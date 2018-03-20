import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

keras.backend.set_image_data_format('channels_last')


class CapsNet:
    def __init__(self, input_shape, n_class, load_weights=False):
        self.model_filename = 'capsnet.h5'
        self.construct(input_shape, n_class)
        self.train_model.summary()

        if load_weights:
            self.train_model.load_weights(self.model_filename)

    def construct(self, input_shape, n_class):
        x = keras.layers.Input(shape=input_shape)

        conv = keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
                                   activation='relu')(x)

        primarycaps = primary_capsule_layer(conv, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16)(primarycaps)

        out_caps = Length(name='capsnet')(digitcaps)

        y = keras.layers.Input(shape=(n_class,))
        masked_by_y = Mask()([digitcaps, y])
        masked = Mask()(digitcaps)

        decoder = keras.models.Sequential(name='decoder')
        decoder.add(keras.layers.Dense(512, activation='relu', input_dim=16*n_class))
        decoder.add(keras.layers.Dense(1024, activation='relu'))
        decoder.add(keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(keras.layers.Reshape(target_shape=input_shape, name='out_recon'))

        train_model = keras.models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = keras.models.Model(x, [out_caps, decoder(masked)])

        self.train_model, self.eval_model = train_model, eval_model

    def train(self, data, args):
        model = self.train_model

        (x_train, y_train), (x_test, y_test) = data

        tb = keras.callbacks.TensorBoard(log_dir='./logs',
                                         batch_size=args.batch_size,
                                         histogram_freq=1)
        checkpoint = keras.callbacks.ModelCheckpoint(self.model_filename, monitor='val_capsnet_acc',
                                                     save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

        def margin_loss(y_true, y_pred):
            L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
                0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

            return tf.reduce_mean(tf.reduce_sum(L, 1))

        model.compile(optimizer=keras.optimizers.Adam(lr=args.lr),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., args.lam_recon],
                      metrics={'capsnet': 'accuracy'})

        def train_generator(x, y, batch_size, shift_fraction=0.):
            train_datagen = ImageDataGenerator(width_shift_range=shift_fraction, height_shift_range=shift_fraction)
            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while True:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])

        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=[[x_test, y_test], [y_test, x_test]],
                            callbacks=[tb, checkpoint, lr_decay])

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


class Length(keras.layers.Layer):
    """Computes the length of the input vectors"""
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(keras.layers.Layer):
    """Mask the vectors so that the true label wins out"""
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            # Mask with true label
            inputs, mask = inputs
        else:
            # Mask with magnitude of output if no true label
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.get_shape().as_list()[1], axis=-1)

        masked = keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """Non-linear squash function acting on vectors"""
    epsilon = 1e-7
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon)
    return scale * vectors


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

        self.built = True

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = tf.map_fn(lambda x: keras.backend.batch_dot(x, self.W, [2, 3]),
                                          elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(keras.backend.batch_dot(c, inputs_hat, [2, 2]))

            if i < self.routings - 1:
                b += keras.backend.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def primary_capsule_layer(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """Conv2D layer preceding dynamic routing layer"""
    output = keras.layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size,
                                 strides=strides, padding=padding,
                                 name='primarycap_conv2d')(inputs)
    outputs = keras.layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)

    return keras.layers.Lambda(squash, name='primarycap_squash')(outputs)
