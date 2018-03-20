#!/usr/bin/env python3
import os
import numpy as np
import keras
from keras.datasets import mnist
from capsnet import CapsNet

def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = keras.utils.to_categorical(y_train.astype('float32'))
    y_test = keras.utils.to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

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

    model = CapsNet(input_shape=x_train.shape[1:], 
                    n_class=len(np.unique(np.argmax(y_train, 1))), 
                    routings=args.routings)

    # model.train(data=((x_train, y_train), (x_test, y_test)), args=args)

    accuracy = model.evaluate(data=(x_test, y_test))

    predictions = model.predict(x_test)

    print(accuracy)
    print(predictions)
