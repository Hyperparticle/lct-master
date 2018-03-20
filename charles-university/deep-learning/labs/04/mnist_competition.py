#!/usr/bin/env python3
import os
import numpy as np
import keras
from tensorflow.examples.tutorials import mnist
from capsnet import CapsNet

def load_mnist():
    data = mnist.input_data.read_data_sets('mnist', reshape=False, seed=42)
    gan_data = mnist.input_data.read_data_sets('mnist-gan', reshape=False, seed=42)

    x_train = np.concatenate((data.train.images, data.validation.images, data.test.images,
                              gan_data.train.images))
    y_train = np.concatenate((data.train.labels, data.validation.labels, data.test.labels,
                              gan_data.train.labels))
    
    x_val = gan_data.validation.images
    y_val = gan_data.validation.labels

    x_test = gan_data.test.images

    y_train, y_val = keras.utils.to_categorical(y_train), keras.utils.to_categorical(y_val)

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

    (x_train, y_train), (x_val, y_val), x_test = load_mnist()

    # n_class = 10
    n_class = len(np.unique(np.argmax(y_val, 1)))

    model = CapsNet(input_shape=x_train.shape[1:], 
                    n_class=n_class, 
                    routings=args.routings)

    model.train(data=((x_train, y_train), (x_val, y_val)), args=args)

    accuracy = model.evaluate(data=(x_val, y_val))

    predictions = model.predict(x_test)

    print(accuracy)
    print(predictions)
