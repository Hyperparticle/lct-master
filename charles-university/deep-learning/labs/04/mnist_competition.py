#!/usr/bin/env python3
import os
import numpy as np
from capsnet import CapsNet, train, test, load_mnist

# class Network:
#     WIDTH = 28
#     HEIGHT = 28
#     LABELS = 10
#
#     def __init__(self, threads, seed=42):
#         # Create an empty graph and a session
#         graph = tf.Graph()
#         graph.seed = seed
#         self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
#                                                                        intra_op_parallelism_threads=threads))
#
#     def construct(self, args):
#         with self.session.graph.as_default():
#             # Inputs
#             self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
#             self.labels = tf.placeholder(tf.int64, [None], name="labels")
#             self.is_training = tf.placeholder(tf.bool, [], name="is_training")
#
#             features = self.images
#
#             features = tf.layers.conv2d(features, 10, 3, 2, 'same', activation=None, use_bias=False)
#             features = tf.layers.batch_normalization(features, training=self.is_training)
#             features = tf.nn.relu(features)
#
#             features = tf.layers.max_pooling2d(features, 3, 2)
#
#             features = tf.contrib.layers.flatten(features)
#
#             features = tf.layers.dense(features, 100, activation=tf.nn.relu)
#
#             output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
#             self.predictions = tf.argmax(output_layer, axis=1)
#
#             # Training
#             loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="loss")
#             global_step = tf.train.create_global_step()
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             with tf.control_dependencies(update_ops):
#                 self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")
#
#             # Summaries
#             self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
#             summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
#             self.summaries = {}
#             with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
#                 self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
#                                            tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
#             with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
#                 for dataset in ["dev", "test"]:
#                     self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
#                                                tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]
#
#             # Initialize variables
#             self.session.run(tf.global_variables_initializer())
#             with summary_writer.as_default():
#                 tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
#
#     def train(self, images, labels):
#         self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})
#
#     def evaluate(self, dataset, images, labels):
#         accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.images: images, self.labels: labels, self.is_training: False})
#         return accuracy
#
#     def predict(self, images):
#         return self.session.run(self.predictions, {self.images: images, self.labels: [], self.is_training: False})


if __name__ == "__main__":
    import argparse

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # # Create logdir name
    # args.logdir = "logs/{}-{}-{}".format(
    #     os.path.basename(__file__),
    #     datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    #     ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    # )
    # if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    (x_train, y_train), (x_test, y_test) = load_mnist()

    model, eval_model, manipulate_model = CapsNet(
        input_shape=x_train.shape[1:], n_class=len(np.unique(np.argmax(y_train, 1))), routings=args.routings)
    model.summary()

    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    test(model=eval_model, data=(x_test, y_test))

    # evaluation(model, sv, num_label)

    # # Construct the network
    # network = Network(threads=args.threads)
    # network.construct(args)

    # # Train
    # for i in range(args.epochs):
    #     while mnist_gan.train.epochs_completed == i:
    #         images, labels = mnist_gan.train.next_batch(args.batch_size)
    #         network.train(images, labels)
    #         images_orig, labels_orig = mnist_orig.train.next_batch(args.batch_size)
    #         network.train(images_orig, labels_orig)

    #     network.evaluate("dev", mnist_gan.validation.images, mnist_gan.validation.labels)

    # accuracy = network.evaluate("test", mnist_gan.test.images, mnist_gan.test.labels)
    # print("{:.2f}".format(100 * accuracy))

    # Compute test_labels, as numbers 0-9, corresponding to mnist_gan.test.images
    # test_labels = network.predict(mnist_gan.test.images)

    # for label in test_labels:
    #     print(label)
