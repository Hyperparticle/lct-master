#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from resnet import resnet


class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._voxels = data["voxels"]
        self._labels = data["labels"] if "labels" in data else None

        # Normalize voxels
        # self._voxels = (self._voxels - self._voxels.mean(axis=0)) / (self._voxels.std(axis=0))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def batches(self, batch_size, shift_fraction=0.0):
        x, y = self._voxels, self._labels

        while True:
            for i in range(0, len(self._voxels), batch_size):
                yield x[i:i + batch_size], y[i:i + batch_size]


class Network:
    LABELS = 10

    def __init__(self):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name="voxels")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            # Classification Network
            x = resnet(self.voxels, args.residual_depth, self.is_training)
            logits = tf.layers.dense(x, self.LABELS, name='output_label')
            self.predictions = tf.argmax(logits, axis=1)

            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, logits, scope='loss')

            # Compute learning rate with stepwise exponential decay
            global_step = tf.train.create_global_step()
            # decay_steps = args.train_size // args.batch_size
            # decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
            # learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate,
            #                                            staircase=True)

            # Set up optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(args.learning_rate)

                # Apply gradient clipping
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.training = optimizer.apply_gradients(zip(gradients, variables),
                                                          global_step=global_step, name='training')

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries["train"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=500, type=int, help="Number of epochs.")
    parser.add_argument("--modelnet_dim", default=20, type=int, help="Dimension of ModelNet data.")
    parser.add_argument("--train_split", default=0.9, type=float, help="Ratio of examples to use as train.")
    parser.add_argument("--learning_rate", default=0.01)
    parser.add_argument("--learning_rate_final", default=0.0005)
    parser.add_argument("--residual_depth", default=3, type=int, help="Depth of residual layers.")
    parser.add_argument("--shift_fraction", default=0.0, type=float)
    parser.add_argument("--load", action='store_true')
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset("modelnet{}-train.npz".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset("modelnet{}-test.npz".format(args.modelnet_dim))

    args.train_size = len(train.voxels)

    # Construct the network
    network = Network()
    network.construct(args)

    if not args.load:
        best_accuracy = 0

        # Train
        for i in range(args.epochs):
            print('Epoch', i)

            with tqdm(total=len(train.voxels)) as pbar:
                batches = train.batches(args.batch_size, args.shift_fraction)
                steps_per_epoch = len(train.voxels)

                total = 0
                while total < steps_per_epoch:
                    voxels, labels = next(batches)
                    network.train(voxels, labels)

                    pbar.update(len(voxels))
                    total += len(voxels)

            accuracy = network.evaluate("dev", dev.voxels, dev.labels)
            print('Val accuracy', accuracy)

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('3d_recognition/model')

    network.load('3d_recognition/model')
    accuracy = network.evaluate("dev", dev.voxels, dev.labels)
    print('Final accuracy', accuracy)

    labels = network.predict(test.voxels)
    with open("3d_recognition_test.txt", "w") as test_file:
        for label in labels:
            print(label, file=test_file)
