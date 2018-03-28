import tensorflow as tf


def resnet(x, residual_depth, training):
    x = preprocess(x)

    x = tf.layers.conv2d(x, 16, [3, 3], strides=1, padding='SAME', use_bias=False,
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    x = tf.layers.batch_normalization(x, training=training)
    x = swish(x)

    for i in range(residual_depth):
        x = residual_layer(x, 16, training)
        assert x.shape[1:] == [28, 28, 16]

    for i in range(residual_depth):
        x = residual_layer(x, 32, training, downsample=(i == 0))
        assert x.shape[1:] == [14, 14, 32]

    # Classification
    for i in range(residual_depth):
        x = residual_layer(x, 64, training, downsample=(i == 0))
        assert x.shape[1:] == [7, 7, 64]

    # Global average pooling
    x = tf.reduce_mean(x, [1, 2])
    assert x.shape[1:] == [64]

    return x


def residual_layer(x, output_channels, training, downsample=False):
    stride = 2 if downsample else 1

    # First hidden layer
    hidden = tf.layers.conv2d(x, output_channels, [3, 3], strides=stride, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)
    hidden = swish(hidden)

    # Second hidden layer
    hidden = tf.layers.conv2d(hidden, output_channels, [3, 3], strides=1, padding='SAME', use_bias=False,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)

    if downsample:
        x = tf.layers.conv2d(x, output_channels, [1, 1], strides=stride, padding='SAME',
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return swish(hidden + x)


def preprocess(x):
    return x - tf.reduce_mean(x, axis=1, keepdims=True)


def swish(x):
    return tf.multiply(x, tf.nn.sigmoid(x))


def group_norm(x, group=7, epsilon=1e-5):
    """
    https://arxiv.org/abs/1803.08494
    https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ImageNetModels/vgg16.py
    """
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tf.get_variable('beta', [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tf.get_variable('gamma', [chan], initializer=tf.constant_initializer(1.0))
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')
    return tf.reshape(out, orig_shape, name='output')
