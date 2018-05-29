"""
Defines a functions for training a NN.
"""

from data_generator import AudioGenerator
import _pickle as pickle

from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda)
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os

import numpy as np
from keras.callbacks import Callback
from utils import int_sequence_to_text
from wer import wer


class Metrics(Callback):
    def __init__(self, input_to_softmax, audio_gen):
        super().__init__()
        self.input_to_softmax = input_to_softmax

        self.transcr = audio_gen.valid_texts
        self.audio_path = audio_gen.valid_audio_paths
        # self.data_points = np.array([audio_gen.normalize(audio_gen.featurize(ap)) for ap in self.audio_path])
        self.data_points, *_ = audio_gen.get_data('valid')

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.input_to_softmax.predict(self.data_points)
        output_lengths = np.array([self.input_to_softmax.output_length(data_point.shape[0]) for data_point in self.data_points])
        pred_ints = K.eval(K.ctc_decode(predictions, output_lengths)[0][0]) + 1

        pred = [''.join(int_sequence_to_text([x for x in p if x != 0])) for p in pred_ints]

        error = [wer(r.split(), h.split()) for r, h in zip(self.transcr, pred)]
        mean_error = np.mean(error)

        print(" - val_wer: {:.2f}".format(mean_error))


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model


def train_model(input_to_softmax, 
                pickle_path,
                save_model_path,
                train_json='train_corpus.json',
                valid_json='valid_corpus.json',
                minibatch_size=20,
                spectrogram=True,
                mfcc_dim=13,
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                epochs=20,
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):
    
    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size,
                               spectrogram=spectrogram,
                               mfcc_dim=mfcc_dim,
                               max_duration=max_duration,
                               sort_by_duration=sort_by_duration)
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # add CTC loss to the NN specified in input_to_softmax
    model = add_ctc_loss(input_to_softmax)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                  optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add callbacks
    callbacks = [
        ModelCheckpoint(filepath='results/' + save_model_path, verbose=0),
        Metrics(input_to_softmax, audio_gen)
    ]

    # train the model
    hist = model.fit_generator(generator=audio_gen.next_train(),
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               validation_data=audio_gen.next_valid(),
                               validation_steps=validation_steps,
                               callbacks=callbacks,
                               verbose=verbose)

    # save model loss
    with open('results/'+pickle_path, 'wb') as f:
        pickle.dump(hist.history, f)