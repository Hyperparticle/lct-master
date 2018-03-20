import numpy as np
import tensorflow as tf
import math
from capsule_layer import CapsuleLayer, PrimaryCap, Length, Mask
import keras
from keras.preprocessing.image import ImageDataGenerator

keras.backend.set_image_data_format('channels_last')

class CapsNet:
    def __init__(self, input_shape, n_class, routings):
        self.construct(input_shape, n_class, routings)
        self.train_model.summary()

    def construct(self, input_shape, n_class, routings):
        """
        A Capsule Network on MNIST.
        :param input_shape: data shape, 3d, [width, height, channels]
        :param n_class: number of classes
        :param routings: number of routing iterations
        :return: Two Keras Models, the first one used for training, and the second one for evaluation.
                `eval_model` can also be used for training.
        """
        x = keras.layers.Input(shape=input_shape)

        # Layer 1: Just a conventional Conv2D layer
        conv1 = keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                name='digitcaps')(primarycaps)

        # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
        # If using tensorflow, this will not be necessary. :)
        out_caps = Length(name='capsnet')(digitcaps)

        # Decoder network.
        y = keras.layers.Input(shape=(n_class,))
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = keras.models.Sequential(name='decoder')
        decoder.add(keras.layers.Dense(512, activation='relu', input_dim=16*n_class))
        decoder.add(keras.layers.Dense(1024, activation='relu'))
        decoder.add(keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(keras.layers.Reshape(target_shape=input_shape, name='out_recon'))

        # Models for training and evaluation (prediction)
        train_model = keras.models.Model([x, y], [out_caps, decoder(masked_by_y)])
        eval_model = keras.models.Model(x, [out_caps, decoder(masked)])

        # manipulate model
        noise = keras.layers.Input(shape=(n_class, 16))
        noised_digitcaps = keras.layers.Add()([digitcaps, noise])
        masked_noised_y = Mask()([noised_digitcaps, y])
        manipulate_model = keras.models.Model([x, y, noise], decoder(masked_noised_y))
        self.train_model, self.eval_model, self.manipulate_model = train_model, eval_model, manipulate_model

    def train(self, data, args):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
        :param args: arguments
        :return: The trained model
        """
        model = self.train_model

        # unpacking the data
        (x_train, y_train), (x_test, y_test) = data

        # callbacks
        tb = keras.callbacks.TensorBoard(log_dir=args.save_dir + '/logs',
                                            batch_size=args.batch_size, histogram_freq=int(args.debug))
        checkpoint = keras.callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                                        monitor='val_capsnet_acc',
                                                        save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

        def margin_loss(y_true, y_pred):
            """
            Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
            :param y_true: [None, n_classes]
            :param y_pred: [None, num_capsule]
            :return: a scalar loss value.
            """
            L = y_true * keras.backend.square(keras.backend.maximum(0., 0.9 - y_pred)) + \
                0.5 * (1 - y_true) * keras.backend.square(keras.backend.maximum(0., y_pred - 0.1))

            return keras.backend.mean(keras.backend.sum(L, 1))

        # compile the model
        model.compile(optimizer=keras.optimizers.Adam(lr=args.lr),
                    loss=[margin_loss, 'mse'],
                    loss_weights=[1., args.lam_recon],
                    metrics={'capsnet': 'accuracy'})

        """
        # Training without data augmentation:
        model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
                validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
        """

        # Begin: Training with data augmentation ---------------------------------------------------------------------#
        def train_generator(x, y, batch_size, shift_fraction=0.):
            train_datagen = ImageDataGenerator(
                width_shift_range=shift_fraction, height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
            generator = train_datagen.flow(x, y, batch_size=batch_size)
            while 1:
                x_batch, y_batch = generator.next()
                yield ([x_batch, y_batch], [y_batch, x_batch])

        # Training with data augmentation. If shift_fraction=0., also no augmentation.
        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=[[x_test, y_test], [y_test, x_test]],
                            callbacks=[tb, checkpoint, lr_decay])
        # End: Training with data augmentation -----------------------------------------------------------------------#

        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

        return model

    def evaluate(self, data):
        model = self.eval_model

        x_test, y_test = data
        y_pred = self.predict(x_test)

        accuracy = np.sum(y_pred == np.argmax(y_test, 1)) / y_test.shape[0]
        return accuracy

    def predict(self, x_test):
        model = self.eval_model

        y_pred, _ = model.predict(x_test, batch_size=64)
        return np.argmax(y_pred, 1)
