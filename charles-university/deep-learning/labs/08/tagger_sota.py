#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tnrange

import morpho_dataset


def dense_to_sparse(dense_tensor, sequence_length):
    # https://github.com/tensorflow/tensorflow/issues/15985
    indices = tf.where(tf.sequence_mask(sequence_length))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=tf.int64)
    return tf.SparseTensor(indices, values, shape)


class Network:
    def __init__(self):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

    def construct(self, args, num_words, num_chars, num_tags):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
            self.charseqs = tf.placeholder(tf.int32, [None, None], name="charseqs")
            self.charseq_lens = tf.placeholder(tf.int32, [None], name="charseq_lens")
            self.charseq_ids = tf.placeholder(tf.int32, [None, None], name="charseq_ids")
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")

            # Generate character embeddings for num_chars of dimensionality args.cle_dim.
            char_embeddings = tf.get_variable("char_embeddings", [num_chars, args.cle_dim])

            # Embed self.charseqs using the character embeddings.
            # [batch, sentence, word, char embed dim]
            embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)

            # Use `tf.nn.bidirectional_dynamic_rnn` to process embedded self.charseqs using
            # a cell of dimensionality `args.cle_dim`.
            fwd_cle = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim)
            bwd_cle = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim)
            char_outputs, __ = tf.nn.bidirectional_dynamic_rnn(fwd_cle, bwd_cle, embedded_chars,
                                                               sequence_length=self.charseq_lens,
                                                               dtype=tf.float32,
                                                               scope='CharBiRNN')

            # Sum the resulting fwd and bwd state to generate character-level word embedding (CLE).
            fwd_bwd = tf.concat(char_outputs, axis=-1)
            cle_table = tf.reduce_sum(fwd_bwd, axis=1)

            # For each word, use suitable CLE according to self.charseq_ids.
            embedded_char_ids = tf.nn.embedding_lookup(cle_table, self.charseq_ids)

            # Create word embeddings for num_words of dimensionality args.we_dim.
            word_embeddings = tf.get_variable("word_embeddings", [num_words, args.we_dim])

            # Embed self.word_ids using the word embeddings.
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            total_word_embeddings = tf.concat([embedded_word_ids, embedded_char_ids], axis=-1)

            # Using tf.nn.bidirectional_dynamic_rnn, process the embedded inputs.
            # Use given rnn_cell (different for fwd and bwd direction).
            fwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim)
            bwd = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_cell_dim)
            outputs, __ = tf.nn.bidirectional_dynamic_rnn(fwd, bwd, total_word_embeddings,
                                                          sequence_length=self.sentence_lens,
                                                          dtype=tf.float32,
                                                          scope='WordBiRNN')

            # Concatenate the outputs for fwd and bwd directions.
            rnn_outputs = tf.concat(outputs, axis=-1)

            # Add a dense layer (without activation) into num_tags classes and
            # store result in `output_layer`.
            output_layer = tf.layers.dense(rnn_outputs, num_tags)

            # log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(output_layer,
            #                                                                       self.tags,
            #                                                                       self.sentence_lens)
            # loss = tf.reduce_mean(-log_likelihood)
            #
            # viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(output_layer,
            #                                                             transition_params,
            #                                                             self.sentence_lens)
            #
            # self.predictions = viterbi_sequence

            # Generate `self.predictions`.
            self.predictions = tf.argmax(output_layer, axis=-1)

            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Training

            # Define `loss` using `tf.losses.sparse_softmax_cross_entropy`, but additionally
            # use `weights` parameter to mask-out invalid words.
            loss = tf.losses.sparse_softmax_cross_entropy(self.tags, output_layer, weights=weights)

            global_step = tf.train.create_global_step()
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            self.training = optimizer.minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train_epoch(self, train, batch_size):
        with tqdm(total=len(train.sentence_lens)) as pbar:
            while not train.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
                self.session.run(self.reset_metrics)
                self.session.run([self.training, self.summaries["train"]],
                                 {self.sentence_lens: sentence_lens,
                                  self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                  self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                                  self.tags: word_ids[train.TAGS]})
                pbar.update(len(sentence_lens))

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        self.session.run(self.reset_metrics)
        labels = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size,
                                                                                              including_charseqs=True)
            predictions = self.session.run([self.predictions],
                             {self.sentence_lens: sentence_lens,
                              self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              self.tags: word_ids[train.TAGS]})
            labels.append(predictions)
        return labels

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--cnne_filters", default=32, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=8, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--we_dim", default=256, type=int, help="Word embedding dimension.")
    parser.add_argument("--load", action='store_true')
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    train = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-train.txt")
    dev = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-dev.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-test.txt", train=train, shuffle_batches=False)

    # Construct the network
    network = Network()
    network.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      len(train.factors[train.TAGS].words))

    if not args.load:
        best_accuracy = 0

        # Train
        for i in range(args.epochs):
            print('Epoch', i)
            network.train_epoch(train, args.batch_size)

            accuracy = network.evaluate("dev", dev, args.batch_size)
            print("{:.2f}".format(100 * accuracy))

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('czech-pdt/model')

    network.load('czech-pdt/model')
    accuracy = network.evaluate("dev", dev, args.batch_size)
    print('Final accuracy', accuracy)

    # Predict test data
    with open("czech_pdt_test.txt", "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
