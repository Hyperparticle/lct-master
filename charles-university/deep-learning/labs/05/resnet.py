import tensorflow as tf


def resnet(x, residual_depth, training):
    """Residual convolutional neural network with global average pooling"""

    x = preprocess(x)

    x = tf.layers.conv2d(x, 32, [3, 3], strides=1, padding='SAME', use_bias=False,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.swish(x)

    for i in range(residual_depth):
        x = residual_layer(x, 32, training)
        assert x.shape[1:] == [28, 28, 32], x.shape[1:]

    for i in range(residual_depth):
        x = residual_layer(x, 64, training, downsample=(i == 0))
        assert x.shape[1:] == [14, 14, 64], x.shape[1:]

    for i in range(residual_depth):
        x = residual_layer(x, 128, training, downsample=(i == 0))
        assert x.shape[1:] == [7, 7, 128], x.shape[1:]

    # Global average pooling
    x = tf.layers.average_pooling2d(x, 7, 1)
    x = tf.layers.flatten(x)
    assert x.shape[1:] == [128], x.shape[1:]

    return x


def residual_layer(x, output_channels, training, downsample=False):
    stride = 2 if downsample else 1

    # First hidden layer
    hidden = tf.layers.conv2d(x, output_channels, [3, 3], strides=stride, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)
    hidden = tf.nn.swish(hidden)

    # Second hidden layer
    hidden = tf.layers.conv2d(hidden, output_channels, [3, 3], strides=1, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)

    if downsample:
        x = tf.layers.conv2d(x, output_channels, [1, 1], strides=stride, padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return tf.nn.swish(hidden + x)


def preprocess(x):
    return x - tf.reduce_mean(x, axis=1, keepdims=True)
