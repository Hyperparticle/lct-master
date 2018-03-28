#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from resnet import residual_layer, preprocess


class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

        self._permutation = np.random.permutation(len(self._images))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm], self._masks[batch_perm]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images))
            return True
        return False


class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="images")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")
            self.masks = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="masks")
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.labels_predictions` of shape [None] and type tf.int64
            # - mask predictions are stored in `self.masks_predictions` of shape [None, 28, 28, 1] and type tf.float32
            #   with values 0 or 1

            x = preprocess(self.images)

            x = tf.layers.conv2d(x, 16, [3,3], strides=1, padding='SAME', use_bias=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.nn.relu(x)

            # [28, 28, 16]
            for i in range(args.residual_depth):
                x = residual_layer(x, 16, self.is_training)
                assert x.shape[1:] == [28, 28, 16]

            # [14, 14, 32]
            for i in range(args.residual_depth):
                x = residual_layer(x, 32, self.is_training, downsample=(i==0))
                assert x.shape[1:] == [14, 14, 32]

            # [7, 7, 64]
            for i in range(args.residual_depth):
                x = residual_layer(x, 64, self.is_training, downsample=(i==0))
                assert x.shape[1:] == [7, 7, 64]

            # Global average pooling
            x = tf.reduce_mean(x, [1,2])
            assert x.shape[1:] == [64]
            # [64]

            output_label = tf.layers.dense(x, self.LABELS, activation=None, name='output_label')
            self.labels_predictions = tf.argmax(output_label, axis=1)

            # x_mask = tf.layers.dense(x, self.WIDTH * self.HEIGHT * 10)
            # output_mask = tf.layers.dense(x_mask, self.HEIGHT * self.WIDTH, activation=None, name='output_mask')
            # output_mask = tf.reshape(output_mask, [-1, self.HEIGHT, self.WIDTH, 1])
            output_mask = tf.zeros_like(self.images)
            self.masks_predictions = tf.round(output_mask)

            loss_label = tf.losses.sparse_softmax_cross_entropy(self.labels, output_label, scope='loss')
            loss_mask = tf.losses.mean_squared_error(self.masks, output_mask)

            # loss = loss_label + loss_mask
            loss = loss_label
            global_step = tf.train.create_global_step()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name='training')

            # Summaries
            accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.labels_predictions), tf.float32))
            only_correct_masks = tf.where(tf.equal(self.labels, self.labels_predictions),
                                          self.masks_predictions, tf.zeros_like(self.masks_predictions))
            intersection = tf.reduce_sum(only_correct_masks * self.masks, axis=[1,2,3])
            iou = tf.reduce_mean(
                intersection / (tf.reduce_sum(only_correct_masks, axis=[1,2,3]) + tf.reduce_sum(self.masks, axis=[1,2,3]) - intersection)
            )

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
                                           tf.contrib.summary.scalar("train/accuracy", accuracy),
                                           tf.contrib.summary.scalar("train/iou", iou),
                                           tf.contrib.summary.image("train/images", self.images),
                                           tf.contrib.summary.image("train/masks", self.masks_predictions)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset+"/loss", loss),
                                               tf.contrib.summary.scalar(dataset+"/accuracy", accuracy),
                                               tf.contrib.summary.scalar(dataset+"/iou", iou),
                                               tf.contrib.summary.image(dataset+"/images", self.images),
                                               tf.contrib.summary.image(dataset+"/masks", self.masks_predictions)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        self.session.run(self.summaries[dataset],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
    parser.add_argument("--residual_depth", default=3, type=int, help="Depth of residual layers.")
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
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz")

    # Construct the network
    network = Network()
    network.construct(args)
    
    # Train
    for i in range(args.epochs):
        print('Epoch', i)

        with tqdm(total=len(train._images)) as pbar:
            while not train.epoch_finished():
                images, labels, masks = train.next_batch(args.batch_size)
                network.train(images, labels, masks)
                pbar.update(len(images))
    
        network.evaluate("dev", dev.images, dev.labels, dev.masks)
    
    labels, masks = network.predict(test.images)
    with open("fashion_masks_test.txt", "w") as test_file:
        for i in range(len(labels)):
            print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
