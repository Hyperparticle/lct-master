import tensorflow as tf


def resnet(x, residual_depth, training):
    """Residual convolutional neural network with global average pooling"""

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
    """Residual convolutional layer based on WideResNet https://arxiv.org/pdf/1605.07146v1.pdf"""

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
