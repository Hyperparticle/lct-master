import tensorflow as tf

def residual_layer(x, output_channels, training, downsample=False):
    stride = 2 if downsample else 1

    input_channels = int(x.shape[-1])

    # First hidden layer
    hidden = tf.layers.conv2d(x, output_channels, [3,3], strides=stride, padding='SAME', use_bias=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)
    hidden = tf.nn.relu(hidden)

    # Second hidden layer
    hidden = tf.layers.conv2d(hidden, output_channels, [3,3], strides=1, padding='SAME', use_bias=False, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = tf.layers.batch_normalization(hidden, training=training)

    if downsample:
        x = tf.layers.conv2d(x, output_channels, [1,1], strides=stride, padding='SAME', kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return tf.nn.relu(hidden + x)

def preprocess(x):
    return x - tf.reduce_mean(x, axis=1, keepdims=True)
