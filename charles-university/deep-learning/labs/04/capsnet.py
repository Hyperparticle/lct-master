import numpy as np
import tensorflow as tf
import keras


class CapsNet:
    """Implementation of CapsNet from https://arxiv.org/pdf/1710.09829.pdf"""

    def __init__(self, input_shape, n_class, load_weights=False):
        self.model_filename = 'capsnet.h5'
        self.construct(input_shape, n_class)
        self.train_model.summary()

        if load_weights:
            self.train_model.load_weights(self.model_filename)

    def construct(self, input_shape, classes):
        x = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv2D(256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)
        primary_caps = keras.layers.Conv2D(8 * 32, kernel_size=9, strides=2, padding='valid')(conv1)
        primary_caps = keras.layers.Reshape([-1, 8])(primary_caps)
        primary_caps = keras.layers.Lambda(squash)(primary_caps)

        caps = CapsuleLayer(classes, 16)(primary_caps)

        output_caps = keras.layers.Lambda(
            lambda inputs: tf.sqrt(tf.reduce_sum(tf.square(inputs), -1)),
            name='capsnet')(caps)

        y = keras.layers.Input((classes,))
        masked_train = Mask()([caps, y])
        masked = Mask()(caps)

        decoder = keras.models.Sequential()
        decoder.add(keras.layers.Dense(512, activation='relu', input_dim=16 * classes))
        decoder.add(keras.layers.Dense(1024, activation='relu'))
        decoder.add(keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(keras.layers.Reshape(target_shape=input_shape, name='reconstruction'))

        train_model = keras.models.Model([x, y], [output_caps, decoder(masked_train)])
        eval_model = keras.models.Model(x, [output_caps, decoder(masked)])

        self.train_model, self.eval_model = train_model, eval_model

    def train(self, data, args):
        model = self.train_model

        (x_train, y_train), (x_test, y_test) = data

        tb = keras.callbacks.TensorBoard(log_dir='./logs',
                                         batch_size=args.batch_size,
                                         histogram_freq=1)
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_filename, monitor='val_capsnet_acc',
                                                     save_best_only=True, save_weights_only=True, verbose=1)
        learn_rate_decay = keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch: args.learn_rate * (args.learn_rate_decay ** epoch))

        margin_loss = lambda true, pred: tf.reduce_mean(tf.reduce_sum(
            true * tf.square(tf.maximum(0., 0.9 - pred)) +
            0.5 * (1 - true) * tf.square(tf.maximum(0., pred - 0.1)), 1))

        model.compile(optimizer=keras.optimizers.Adam(lr=args.learn_rate),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., 0.392],
                      metrics={'capsnet': 'accuracy'})

        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, 0.1),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=[[x_test, y_test], [y_test, x_test]],
                            callbacks=[tb, checkpoint, learn_rate_decay])

        model.save_weights(self.model_filename)

        return model

    def evaluate(self, data):
        x_test, y_test = data
        y_pred = self.predict(x_test)

        accuracy = np.sum(y_pred == np.argmax(y_test, 1)) / y_test.shape[0]
        return accuracy

    def predict(self, x_test):
        model = self.eval_model

        y_pred, _ = model.predict(x_test)
        return np.argmax(y_pred, 1)


class CapsuleLayer(keras.layers.Layer):
    """Capsule layer with vector outputs (instead of traditional scalar values)"""
    def __init__(self, num_capsule, dim_capsule,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = 3
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer, name='W')

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = tf.map_fn(lambda x:  tf.keras.backend.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.keras.backend.batch_dot(c, inputs_hat, [2, 2]))

            if i < self.routings - 1:
                b += tf.keras.backend.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


class Mask(keras.layers.Layer):
    """Mask the vectors so that the true label wins out"""
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(tf.argmax(x, 1), x.get_shape().as_list()[1], axis=-1)

        masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """Non-linear squash function acting on vectors"""
    epsilon = 1e-7
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon)
    return scale * vectors


def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=shift_fraction, height_shift_range=shift_fraction)
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while True:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
