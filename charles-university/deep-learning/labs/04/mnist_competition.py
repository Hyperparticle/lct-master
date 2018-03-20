#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from capsnet import CapsNet


def load_mnist():
    gan_data = mnist.input_data.read_data_sets('mnist-gan', reshape=False, seed=42)

    x_train = gan_data.train.images
    y_train = tf.keras.utils.to_categorical(gan_data.train.labels)

    x_val = gan_data.validation.images
    y_val = tf.keras.utils.to_categorical(gan_data.validation.labels)

    x_test = gan_data.test.images

    return (x_train, y_train), (x_val, y_val), x_test


if __name__ == "__main__":
    import argparse

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learn_rate', default=0.001, type=float)
    parser.add_argument('--learn_rate_decay', default=0.9, type=float)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), x_test = load_mnist()

    model = CapsNet(x_train.shape[1:], 10, args.load)

    if not args.load:
        model.train(data=((x_train, y_train), (x_val, y_val)), args=args)

    accuracy = model.evaluate(data=(x_val, y_val))
    predictions = model.predict(x_test)

    print(accuracy, '\n')

    for label in predictions:
        print(label)
