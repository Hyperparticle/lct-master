#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import skopt

# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding='utf-8') as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item:item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False

class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, logdir, activations):
        with self.session.graph.as_default():
            # Inputs
            self.windows = tf.placeholder(tf.int32, [None, 2 * args.window + 1], name="windows")
            self.labels = tf.placeholder(tf.int64, [None], name="labels") # Or you can use tf.int32
            self.is_training = tf.placeholder_with_default(False, [], name="is_training")

            # Define a suitable network with appropriate loss function
            hidden_layer = tf.layers.flatten(tf.one_hot(self.windows, args.alphabet_size))
            for _ in range(args.num_dense_layers):
                hidden_layer = tf.layers.dense(hidden_layer, args.num_dense_nodes, activation=activations[args.activation])
                hidden_layer = tf.contrib.layers.layer_norm(hidden_layer)
                hidden_layer = tf.layers.dropout(hidden_layer, args.dropout, training=self.is_training)

            output_layer = tf.layers.dense(hidden_layer, 2, activation=None, name="output_layer")

            self.predictions = tf.argmax(output_layer, axis=1)

            # Define training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Construct the saver
            tf.add_to_collection("end_points/windows", self.windows)
            tf.add_to_collection("end_points/labels", self.labels)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]], {self.windows: windows, self.labels: labels, self.is_training: True})

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        # Load the metagraph
        with self.session.graph.as_default():
            self.saver = tf.train.import_meta_graph(path + ".meta")

            # Attach the end points
            self.observations = tf.get_collection("end_points/windows")[0]
            self.actions = tf.get_collection("end_points/labels")[0]

    def evaluate(self, dataset, windows, labels):
        acc, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.windows: windows, self.labels: labels})
        return acc

    def predict(self, windows):
        return self.session.run(self.predictions, {self.windows: windows, self.labels: []})

def test_network(train, dev, args, logdir, activations):
    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, logdir, activations)

    # Train
    for i in range(args.epochs):
        print('Epoch:', i+1, 'of', args.epochs)
        with tqdm(total=len(train._permutation)) as pbar:
            while not train.epoch_finished():
                windows, labels = train.next_batch(args.batch_size)
                network.train(windows, labels)
                pbar.update(len(windows))

        dev_windows, dev_labels = dev.all_data()
        network.evaluate("dev", dev_windows, dev_labels)
    accuracy = network.evaluate("dev", dev_windows, dev_labels)
    return network, accuracy

def fitness(x):
    global call_num, best_accuracy, activations

    args.learning_rate, args.num_dense_layers, args.num_dense_nodes, args.dropout, args.window, args.activation = x

    call_num += 1

    print('Iteration', call_num)

    # Create logdir name
    logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("uppercase_data_test.txt", args.window, alphabet=train.alphabet)

    try:
        network, accuracy = test_network(train, dev, args, logdir, activations)
    except tf.errors.ResourceExhaustedError:
        return 1.0

    if accuracy > best_accuracy:
        print()
        print('New best')
        print('Reward: {:.2f}'.format(accuracy))
        print('learning rate: {0:.1e}'.format(args.learning_rate))
        print('num_dense_layers:', args.num_dense_layers)
        print('num_dense_nodes:', args.num_dense_nodes)
        print('num_epochs:', args.epochs)
        print('dropout: {0:.1e}'.format(args.dropout))
        print('window:', args.window)
        print('activation:', args.activation)

        network.save('upperacase/model')
        best_accuracy = accuracy

    network.session.close()
    del network

    return 1.0 - accuracy

if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphabet_size", default=200, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
    parser.add_argument("--iter", default=1000, type=int, help="Number of iterations.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=10, type=int, help="Size of the window to use.")

    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--num_dense_layers", default=2, type=int)
    parser.add_argument("--num_dense_nodes", default=300, type=int)

    args = parser.parse_args()

    activations = {
        'tanh': tf.nn.tanh,
        'relu': tf.nn.relu,
        'elu': tf.nn.elu,
        'selu': tf.nn.selu
    }

    dim_learning_rate = skopt.space.Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_num_dense_layers = skopt.space.Integer(low=1, high=6, name='num_dense_layers')
    dim_num_dense_nodes = skopt.space.Integer(low=200, high=4096, name='num_dense_nodes')
    # dim_num_epochs = skopt.space.Integer(low=1, high=5, name='num_epochs')
    dim_dropout = skopt.space.Real(low=0.0, high=0.4, name='dropout')
    dim_window = skopt.space.Integer(low=5, high=20, name='window')
    dim_activation = skopt.space.Categorical(activations.keys(), name='activation')

    dimensions = [dim_learning_rate,
                  dim_num_dense_layers,
                  dim_num_dense_nodes,
                  dim_dropout,
                  dim_window,
                  dim_activation]
    default_parameters = [0.00045, 3, 1024, 0.25, 8, 'elu']

    best_accuracy = 0.0
    call_num = 0

    res_gp = skopt.gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=args.iter,
                            x0=default_parameters)
    
    print("Best score=%.4f" % res_gp.fun)
    print("""Best parameters:
            - learning_rate=%d
            - num_dense_layers=%.6f
            - num_dense_nodes=%d
            - num_epochs=%d
            - dropout=%d""" % (res_gp.x[0], res_gp.x[1], 
                                        res_gp.x[2], res_gp.x[3], 
                                        res_gp.x[4]))
    
    # # Generate the uppercased test set
    # test_windows, _ = test.all_data()
    # predictions = network.predict(test_windows)
    # text = ''.join(c.upper() if p == 1 else c for c,p in zip(test.text, predictions))

    # print(text)


    # learning rate: 4.6e-04
    # num_dense_layers: 3
    # num_dense_nodes: 995
    # num_epochs: 5
    # dropout: 2.5e-01
    # window: 6
    # activation: relu