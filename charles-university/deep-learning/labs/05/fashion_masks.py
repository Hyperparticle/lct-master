#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from resnet import resnet, swish


def iou_accuracy(pred_labels, pred_masks, gold_labels, gold_masks):
    iou = 0
    for pred_label, pred_mask, gold_label, gold_mask in zip(pred_labels, pred_masks, gold_labels, gold_masks):
        if pred_label == gold_label:
            system_mask = np.array(pred_mask, dtype=gold_mask.dtype).reshape(gold_mask.shape)
            system_pixels = np.sum(system_mask)
            gold_pixels = np.sum(gold_mask)
            intersection_pixels = np.sum(system_mask * gold_mask)
            iou += intersection_pixels / (system_pixels + gold_pixels - intersection_pixels)
    return 100 * iou / len(gold_labels)


class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._images = data["images"]
        self._labels = data["labels"] if "labels" in data else None
        self._masks = data["masks"] if "masks" in data else None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def masks(self):
        return self._masks

    def batches(self, batch_size, shift_fraction=0., seed=42):
        x, y, m = self._images, self._labels, self._masks

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=shift_fraction, height_shift_range=shift_fraction)
        gen1 = train_datagen.flow(x, y, batch_size=batch_size, seed=seed)
        gen2 = train_datagen.flow(m, batch_size=batch_size, seed=seed)

        while True:
            x_batch, y_batch = gen1.next()
            m_batch = gen2.next()

            yield x_batch, y_batch, m_batch


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

            # Classification
            x_label = resnet(self.images, args.residual_depth, self.is_training)
            output_label = tf.layers.dense(x_label, self.LABELS, name='output_label')
            self.labels_predictions = tf.argmax(output_label, axis=1)

            # Segmentation
            x_mask = resnet(self.images, args.residual_depth, self.is_training)
            x_mask = tf.layers.dense(x_mask, 512, activation=swish)
            x_mask = tf.layers.dense(x_mask, 1024, activation=swish)
            x_mask = tf.layers.dense(x_mask, self.HEIGHT * self.WIDTH, activation=None, name='output_mask')
            output_mask = tf.reshape(x_mask, [-1, self.HEIGHT, self.WIDTH, 1])
            self.masks_predictions = tf.round(output_mask)

            loss_label = tf.losses.sparse_softmax_cross_entropy(self.labels, output_label, scope='loss')
            loss_mask = tf.losses.mean_squared_error(self.masks, output_mask)

            loss = loss_label + loss_mask
            global_step = tf.train.create_global_step()

            # Compute learning rate with decay
            decay_steps = args.train_size // args.batch_size
            decay_rate = (args.learning_rate_final / args.learning_rate) ** (1 / (args.epochs - 1))
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, decay_steps, decay_rate,
                                                       staircase=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.training = optimizer.apply_gradients(zip(gradients, variables),
                                                          global_step=global_step, name='training')

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

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, images, labels, masks):
        self.session.run([self.training, self.summaries["train"]],
                         {self.images: images, self.labels: labels, self.masks: masks, self.is_training: True})

    def evaluate(self, dataset, images, labels, masks):
        pred_labels, pred_masks, _ = self.session.run(
            [self.labels_predictions, self.masks_predictions, self.summaries[dataset]],
            {self.images: images, self.labels: labels, self.masks: masks, self.is_training: False})
        return iou_accuracy(pred_labels, pred_masks, labels, masks)

    def predict(self, images):
        return self.session.run([self.labels_predictions, self.masks_predictions],
                                {self.images: images, self.is_training: False})

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--learning_rate_final", default=0.0001)
    parser.add_argument("--residual_depth", default=3, type=int, help="Depth of residual layers.")
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
    train = Dataset("fashion-masks-train.npz")
    dev = Dataset("fashion-masks-dev.npz")
    test = Dataset("fashion-masks-test.npz")

    args.train_size = len(train.images)

    # Construct the network
    network = Network()
    network.construct(args)

    if not args.load:
        best_accuracy = 0

        # Train
        for i in range(args.epochs):
            print('Epoch', i)

            with tqdm(total=len(train.images)) as pbar:
                batches = train.batches(args.batch_size)
                steps_per_epoch = len(train.images)
                total = 0
                while total < steps_per_epoch:
                    images, labels, masks = next(batches)
                    network.train(images, labels, masks)
                    pbar.update(len(images))
                    total += len(images)

            accuracy = network.evaluate("dev", dev.images, dev.labels, dev.masks)
            print('Val accuracy', accuracy)
            if accuracy > best_accuracy:
                network.save('fashion_masks/model')

    network.load('fashion_masks/model')
    accuracy = network.evaluate("dev", dev.images, dev.labels, dev.masks)
    print('Final accuracy', accuracy)

    labels, masks = network.predict(test.images)
    with open("fashion_masks_test.txt", "w") as test_file:
        for i in range(len(labels)):
            print(labels[i], *masks[i].astype(np.uint8).flatten(), file=test_file)
