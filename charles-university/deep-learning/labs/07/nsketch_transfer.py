#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
            len(self._images))
        # Normalize images
        # self._images = (self._images - self._images.mean(axis=0)) / (self._images.std(axis=0))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
                len(self._images))
            return True
        return False

    def batches(self, batch_size, shift_fraction=0.):
        x, y = self._images, self._labels

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=shift_fraction, height_shift_range=shift_fraction)
        gen = train_datagen.flow(x, y, batch_size=batch_size)

        while True:
            x_batch, y_batch = gen.next()
            yield x_batch, y_batch


class Network:
    WIDTH, HEIGHT = 224, 224
    CLASSES = 250

    def __init__(self, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph)

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=False)
            self.nasnet_saver = tf.train.Saver()

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            x = features
            x = tf.layers.dense(x, 2048, activation=tf.nn.relu)

            output = tf.layers.dense(x, self.CLASSES)

            self.predictions = tf.argmax(output, axis=1)
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output, scope='loss')

            global_step = tf.train.create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(args.learning_rate)

                # Apply gradient clipping
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.training = optimizer.apply_gradients(zip(gradients, variables),
                                                          global_step=global_step, name='training')

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name="given_loss")
                self.given_accuracy = tf.placeholder(tf.float32, [], name="given_accuracy")
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.given_accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries["train"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.images: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)

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
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--nasnet", default="nets/nasnet/model.ckpt", type=str, help="NASNet checkpoint path.")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--shift_fraction", default=0.0)
    parser.add_argument("--load", action='store_true')
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(vars(args).items()))).replace("/", "-")
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("nsketch/nsketch-train.npz")
    dev = Dataset("nsketch/nsketch-dev.npz", shuffle_batches=False)
    test = Dataset("nsketch/nsketch-test.npz", shuffle_batches=False)

    # Construct the network
    network = Network()
    network.construct(args)

    if not args.load:
        best_accuracy = 0

        # Train
        for i in range(args.epochs):
            print('Epoch', i)

            with tqdm(total=len(train.images)) as pbar:
                batches = train.batches(args.batch_size, args.shift_fraction)
                steps_per_epoch = len(train.images)
                total = 0
                while total < steps_per_epoch:
                    images, labels = next(batches)
                    network.train_batch(images, labels)
                    pbar.update(len(images))
                    total += len(images)

            accuracy = network.evaluate("dev", dev, args.batch_size)
            print('Val accuracy', accuracy)

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('nsketch/model')

    network.load('nsketch/model')
    accuracy = network.evaluate("dev", dev, args.batch_size)
    print('Final accuracy', accuracy)

    # Predict test data
    with open("{}/nsketch_transfer_test.txt".format(args.logdir), "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
