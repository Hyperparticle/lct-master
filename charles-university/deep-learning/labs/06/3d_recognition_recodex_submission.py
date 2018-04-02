# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from resnet import resnet


class Dataset:
    def __init__(self, filename):
        data = np.load(filename)
        self._voxels = data[\"voxels\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

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

    def batches(self, batch_size):
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
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name=\"voxels\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Computation and training.
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
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=64, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=100, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--modelnet_dim\", default=20, type=int, help=\"Dimension of ModelNet data.\")
    parser.add_argument(\"--train_split\", default=0.9, type=float, help=\"Ratio of examples to use as train.\")
    parser.add_argument(\"--learning_rate\", default=0.01)
    parser.add_argument(\"--residual_depth\", default=4, type=int, help=\"Depth of residual layers.\")
    parser.add_argument(\"--load\", action='store_true')
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset(\"modelnet{}-train.npz\".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset(\"modelnet{}-test.npz\".format(args.modelnet_dim))

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
                batches = train.batches(args.batch_size)
                steps_per_epoch = len(train.voxels)

                total = 0
                while total < steps_per_epoch:
                    voxels, labels = next(batches)
                    network.train(voxels, labels)

                    pbar.update(len(voxels))
                    total += len(voxels)

            accuracy = network.evaluate(\"dev\", dev.voxels, dev.labels)
            print('Val accuracy', accuracy)

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('3d_recognition/model')

    network.load('3d_recognition/model')
    accuracy = network.evaluate(\"dev\", dev.voxels, dev.labels)
    print('Final accuracy', accuracy)

    labels = network.predict(test.voxels)
    with open(\"3d_recognition_test.txt\", \"w\") as test_file:
        for label in labels:
            print(label, file=test_file)
"""

source_2 = """import tensorflow as tf


def resnet(x, residual_depth, training):
    \"\"\"Residual convolutional neural network with global average pooling\"\"\"

    x = tf.layers.conv3d(x, 16, [3, 3, 3], strides=1, padding='SAME', use_bias=False,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.layers.batch_normalization(x, training=training, momentum=0.9, epsilon=1e-5)
    x = tf.nn.swish(x)
    assert x.shape[-1] == 16, x.shape[1:]

    for i in range(residual_depth):
        x = residual_layer(x, 16, training)
        assert x.shape[-1] == 16, x.shape[1:]

    for i in range(residual_depth):
        x = residual_layer(x, 32, training, downsample=(i == 0))
        assert x.shape[-1] == 32, x.shape[1:]

    for i in range(residual_depth):
        x = residual_layer(x, 64, training, downsample=(i == 0))
        assert x.shape[-1] == 64, x.shape[1:]

    # Global average pooling
    x = tf.reduce_mean(x, [1, 2, 3])
    assert x.shape[-1] == 64, x.shape[1:]

    return x


def residual_layer(x, output_channels, training, downsample=False, weight_decay=0.0005):
    \"\"\"Residual convolutional layer based on WideResNet https://arxiv.org/pdf/1605.07146v1.pdf\"\"\"

    stride = 2 if downsample else 1

    # First hidden layer
    hidden = tf.layers.conv3d(x, output_channels, [3, 3, 3], strides=stride, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    hidden = tf.layers.batch_normalization(hidden, training=training, momentum=0.9, epsilon=1e-5)
    hidden = tf.nn.swish(hidden)

    # Second hidden layer
    hidden = tf.layers.conv3d(hidden, output_channels, [3, 3, 3], strides=1, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    hidden = tf.layers.batch_normalization(hidden, training=training, momentum=0.9, epsilon=1e-5)

    if downsample:
        x = tf.layers.conv3d(x, output_channels, [1, 1, 1], strides=stride, padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

    return tf.nn.swish(hidden + x)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;0G51MqL0Hf`djUAS-MB88cawTn>Q(YAH=xUGgvTt-)4BK{6-p9h5Ced~AwOsiw(`k($@*j{O&bwWsDSI0(0FA(7PE-vP41Vi0}2zCvyb)wP!e^9E@<es7!un1esk*W>~a_-8zjBa6kET{ON)CS56ukiG&`r2?p@rnU?cS|?}5V0V}f<L5a5#!UQT6rqCB0J7*@Q+<JpwU+qUTZN`@)-9LAz1?s4+NNdxsGT|2bbmcu+p=Sj>g43Swn{D7xY2<dYMXp?Ma)Sfd(m#ej|E8Cz4&3c%xh(qc5hs&+<_9IjEe}o{)6U{T!)Ruu3g96q%>R;z|Kn=smeG|mK9q8wh_lwo9Uq%gcI(4^M6rb=snADL0$TjH1d6LKSgB)4*GEb0=Bc1s9$z+Oe*sw_!PM?2s7Vq67wS-NgMd|oV<j1iH9L&8R(D?x#7=azC0#N`ph!f8*=k{4p1u4N-EP?M+S+64Df1dwiQ2s(OMsaSvDHsWN8-jcZba!2TPFWX4MAgvZMataU7<K>l^HE-2AJoh=>v8>uUfVmaiK=H><VgF7K_xQ*_%fw$6>_)KOC`<Tj;ssV%j+4dzS$qk?@M_qE3Xrb?j@tL_PZQj<BU3JC3rvbI{SE*-s>PTL_!DRm1{f0#r0OfuLdF9~GI?FKhntWC8f4+s--+JN--=1XGNhZrWleQLH^p2jCt^BYmpcrRR4;hTAywvx#}R>4Qai{~oxBnXao=D~EG>8+jo)+|Y2MF0Q*d7_$fCWk6W00H6zm<|8{)hsg~vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
