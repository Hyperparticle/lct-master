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
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), x_test = load_mnist()

    model = CapsNet(input_shape=x_train.shape[1:],
                    n_class=len(np.unique(np.argmax(y_val, 1))),
                    load_weights=args.load)

    if not args.load:
        model.train(data=((x_train, y_train), (x_val, y_val)), args=args)

    accuracy = model.evaluate(data=(x_val, y_val))

    predictions = model.predict(x_test)

    print(accuracy, '\n')

    for label in predictions:
        print(label)
