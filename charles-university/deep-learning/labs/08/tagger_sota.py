#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm, tnrange
from modules import *

import morpho_dataset


def learning_rate_scheduler(epoch):
    return args.learning_rate * (2 ** -epoch)


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
            self.learning_rate = tf.placeholder_with_default(0.001, [], name="learning_rate")
            self.is_training = tf.placeholder_with_default(False, [], name="is_training")

            num_blocks = 2
            num_heads = 8

            with tf.variable_scope("word_embedding"):
                # Create word embeddings for num_words of dimensionality args.we_dim.
                word_embeddings = tf.get_variable("word_embeddings", [num_words, args.we_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer())

                # Embed self.word_ids using the word embeddings.
                embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self.word_ids)

            with tf.variable_scope("char_embedding"):
                # Generate character embeddings for num_chars of dimensionality args.cle_dim.
                char_embeddings = tf.get_variable("char_embeddings", [num_chars, args.cle_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer())

                # Embed self.charseqs using the character embeddings.
                # [batch, sentence, word, char embed dim]
                embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.charseqs)
                embedded_chars = tf.layers.dropout(embedded_chars, rate=args.dropout, training=self.is_training)

            with tf.variable_scope("cle_embedding"):
                # Use `tf.nn.bidirectional_dynamic_rnn` to process embedded self.charseqs using
                # a cell of dimensionality `args.cle_dim`.
                fwd_cle = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_char_dim)
                bwd_cle = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_char_dim)
                char_outputs, __ = tf.nn.bidirectional_dynamic_rnn(fwd_cle, bwd_cle, embedded_chars,
                                                                   sequence_length=self.charseq_lens,
                                                                   dtype=tf.float32,
                                                                   scope='CharBiRNN')

                # Sum the resulting fwd and bwd state to generate character-level word embedding (CLE).
                fwd_bwd = tf.concat(char_outputs, axis=-1)
                cle_table = tf.reduce_sum(fwd_bwd, axis=1)

                # For each word, use suitable CLE according to self.charseq_ids.
                embedded_char_ids_cle = tf.nn.embedding_lookup(cle_table, self.charseq_ids)
                embedded_char_ids_cle = tf.layers.dropout(embedded_char_ids_cle, rate=args.dropout,
                                                          training=self.is_training)

            total_word_embeddings = tf.concat([embedded_word_ids, embedded_char_ids_cle], axis=-1)

            ## Positional Encoding
            total_word_embeddings += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(total_word_embeddings)[1]), 0),
                                [tf.shape(total_word_embeddings)[0], 1]),
                        vocab_size=num_words,
                        num_units=total_word_embeddings.shape[-1],
                        zero_pad=False,
                        scale=False,
                        scope="enc_pe")
            # self.enc += positional_encoding(self.word_ids,
            #                                 num_units=hp.hidden_units,
            #                                 zero_pad=False,
            #                                 scale=False,
            #                                 scope="enc_pe")

            total_word_embeddings = tf.layers.dropout(total_word_embeddings, rate=args.dropout, training=self.is_training)

            # Encoder
            with tf.variable_scope("encoder"):
                enc_hidden_units = total_word_embeddings.shape[-1]
                self.enc = total_word_embeddings

                ## Blocks
                for i in range(num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=enc_hidden_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=args.dropout,
                                                       is_training=self.is_training,
                                                       causality=False)

                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4 * enc_hidden_units.value, enc_hidden_units])

            # Add a dense layer (without activation) into num_tags classes
            output_layer = tf.layers.dense(self.enc, num_tags)

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

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.learning_rate)

                # Apply gradient clipping
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy),
                                           tf.contrib.summary.scalar("train/learning_rate", self.learning_rate)]
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

    def train_epoch(self, train, dev, batch_size, learning_rate):
        global best_accuracy

        with tqdm(total=len(train.sentence_lens)) as pbar:
            step = 1
            while not train.epoch_finished():
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = train.next_batch(batch_size, including_charseqs=True)
                self.session.run(self.reset_metrics)
                self.session.run([self.training, self.summaries["train"]],
                                 {self.sentence_lens: sentence_lens,
                                  self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                  self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                                  self.tags: word_ids[train.TAGS],
                                  self.learning_rate: learning_rate,
                                  self.is_training: True})
                pbar.update(len(sentence_lens))

                if step % 1000 == 0:
                    accuracy = network.evaluate("dev", dev, args.batch_size)

                    print("{:.2f}".format(100 * accuracy))

                    if accuracy > best_accuracy:
                        print('^^ New best ^^')
                        best_accuracy = accuracy
                        network.save('sota/model')

                step += 1

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
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--rnn_char_dim", default=256, type=int, help="RNN cell dimension.")
    parser.add_argument("--rnn_word_dim", default=1200, type=int, help="RNN cell dimension.")
    parser.add_argument("--cle_dim", default=128, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=48, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=3, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
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
            network.train_epoch(train, dev, args.batch_size, learning_rate_scheduler(i))

            accuracy = network.evaluate("dev", dev, args.batch_size)

            print("{:.2f}".format(100 * accuracy))

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('sota/model')

    network.load('sota/model')
    accuracy = network.evaluate("dev", dev, args.batch_size)
    print('Final accuracy', accuracy)

    # Predict test data
    with open("czech_pdt_test.txt", "w") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
