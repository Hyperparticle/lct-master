#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import morpho_dataset


def learning_rate_scheduler(epoch):
    """Outputs the learning rate as a function of the current epoch (2^-n)"""
    # return args.learning_rate * (2 ** -epoch)
    return args.learning_rate


class MorphoAnalyzer:
    """ Loader for data of morphological analyzer.

    The loaded analyzer provides an only method `get(word)` returning
    a list of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    class LemmaTag:
        def __init__(self, lemma, tag):
            self.lemma = lemma
            self.tag = tag

    def __init__(self, filename):
        self.analyses = {}

        with open(filename, "r", encoding="utf-8") as analyzer_file:
            for line in analyzer_file:
                line = line.rstrip("\n")
                columns = line.split("\t")

                analyses = []
                for i in range(1, len(columns) - 1, 2):
                    analyses.append(MorphoAnalyzer.LemmaTag(columns[i], columns[i + 1]))
                self.analyses[columns[0]] = analyses

    def get(self, word):
        return self.analyses.get(word, [])


class Network:
    def __init__(self):
        # Create an empty graph and a session
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

    def construct(self, args, source_chars, target_chars, tag_chars, num_tags, num_words, bow, eow):
        with self.session.graph.as_default():
            # Inputs
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")

            self.source_ids = tf.placeholder(tf.int32, [None, None], name="source_ids")
            self.source_seqs = tf.placeholder(tf.int32, [None, None], name="source_seqs")
            self.source_seq_lens = tf.placeholder(tf.int32, [None], name="source_seq_lens")

            self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")

            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            self.tag_ids = tf.placeholder(tf.int32, [None, None], name="tag_ids")
            self.tag_seqs = tf.placeholder(tf.int32, [None, None], name="tag_seqs")
            self.tag_seq_lens = tf.placeholder(tf.int32, [None], name="tag_seq_lens")

            self.target_ids = tf.placeholder(tf.int32, [None, None], name="target_ids")
            self.target_seqs = tf.placeholder(tf.int32, [None, None], name="target_seqs")
            self.target_seq_lens = tf.placeholder(tf.int32, [None], name="target_seq_lens")

            # Map sentences -> word list
            self.word_indexes = tf.placeholder(tf.int32, [None, 2], name='word_indexes')

            self.learning_rate = tf.placeholder_with_default(0.001, [], name="learning_rate")
            self.is_training = tf.placeholder_with_default(False, [], name="is_training")

            # Append EOW after target_seqs
            target_seqs = tf.reverse_sequence(self.target_seqs, self.target_seq_lens, 1)
            target_seqs = tf.pad(target_seqs, [[0, 0], [1, 0]], constant_values=eow)
            target_seq_lens = self.target_seq_lens + 1
            target_seqs = tf.reverse_sequence(target_seqs, target_seq_lens, 1)

            # Encode the source sequences on the character level (self.source_seqs)
            with tf.variable_scope("encoder"):
                source_embeddings = tf.get_variable("source_embeddings", [source_chars, args.char_dim])

                embedded_source_seqs = tf.nn.embedding_lookup(source_embeddings, self.source_seqs)
                embedded_source_seqs = tf.layers.dropout(embedded_source_seqs,
                                                         rate=args.dropout,
                                                         training=self.is_training)

                source_encoder_outputs, source_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(args.rnn_dim),
                        tf.nn.rnn_cell.GRUCell(args.rnn_dim),
                        embedded_source_seqs,
                        sequence_length=self.source_seq_lens,
                        dtype=tf.float32,
                        scope="source_encoder")
                source_encoder_outputs = tf.reduce_sum(source_encoder_outputs, axis=0)
                source_encoder_state = tf.reduce_sum(source_encoder_state, axis=0)

            # Encode the source tags on the character level (self.tag_seqs)
            # NOTE: Make sure to include valid tags on the source side.
            # Otherwise, the network may have poor accuracy on inference.
            with tf.variable_scope("encoder_tags"):
                tag_embeddings = tf.get_variable("tag_embeddings", [tag_chars, args.tag_char_dim])

                embedded_tag_seqs = tf.nn.embedding_lookup(tag_embeddings, self.tag_seqs)
                embedded_tag_seqs = tf.layers.dropout(embedded_tag_seqs, rate=args.dropout, training=self.is_training)

                tag_encoder_outputs, tag_encoder_states = tf.nn.bidirectional_dynamic_rnn(
                        tf.nn.rnn_cell.GRUCell(args.tag_rnn_dim),
                        tf.nn.rnn_cell.GRUCell(args.tag_rnn_dim),
                        embedded_tag_seqs,
                        sequence_length=self.tag_seq_lens,
                        dtype=tf.float32,
                        scope="tag_encoder")
                tag_encoder_outputs = tf.reduce_sum(tag_encoder_outputs, axis=0)
                tag_encoder_states = tf.reduce_sum(tag_encoder_states, axis=0)

            # with tf.variable_scope("encoder_words"):
            #     # source_embeddings = tf.get_variable("source_embeddings", [source_chars, args.char_dim])
            #     word_embeddings = tf.get_variable("word_embeddings", [num_words, args.rnn_dim])
            #
            #     embedded_word_seqs = tf.nn.embedding_lookup(word_embeddings, self.word_ids)
            #     embedded_word_seqs = tf.layers.dropout(embedded_word_seqs,
            #                                            rate=args.dropout,
            #                                            training=self.is_training)
            #
            #     # Create multiple RNN layers
            #     (fwd_output, bwd_output), (fwd_state, bwd_state) = tf.nn.bidirectional_dynamic_rnn(
            #         tf.nn.rnn_cell.GRUCell(args.rnn_dim),
            #         tf.nn.rnn_cell.GRUCell(args.rnn_dim),
            #         embedded_word_seqs,
            #         sequence_length=self.sentence_lens,
            #         dtype=tf.float32,
            #         scope="word_encoder")
            #     word_encoder_outputs = fwd_output + bwd_output
            #     word_encoder_state = fwd_state + bwd_state  # Keep the last state of the multilayer RNN for decoding
            #     word_encoder_state = tf.layers.dropout(word_encoder_state, rate=args.dropout, training=self.is_training)

            # Mask out sentences using embedding lookups to produce batches of words
            with tf.variable_scope("masks"):
                sentence_mask = tf.sequence_mask(self.sentence_lens)

                source_encoder_outputs = tf.boolean_mask(tf.nn.embedding_lookup(source_encoder_outputs, self.source_ids), sentence_mask)
                source_encoder_state = tf.boolean_mask(tf.nn.embedding_lookup(source_encoder_state, self.source_ids), sentence_mask)
                source_lens = tf.boolean_mask(tf.nn.embedding_lookup(self.source_seq_lens, self.source_ids), sentence_mask)

                tag_encoder_outputs = tf.boolean_mask(tf.nn.embedding_lookup(tag_encoder_outputs, self.tag_ids), sentence_mask)
                tag_encoder_states = tf.boolean_mask(tf.nn.embedding_lookup(tag_encoder_states, self.tag_ids), sentence_mask)

                target_seqs = tf.boolean_mask(tf.nn.embedding_lookup(target_seqs, self.target_ids), sentence_mask)
                target_lens = tf.boolean_mask(tf.nn.embedding_lookup(target_seq_lens, self.target_ids), sentence_mask)

                # word_encoder_state = tf.gather_nd(word_encoder_state, self.word_indexes)
                # word_encoder_outputs = tf.gather_nd(word_encoder_outputs, self.word_indexes)

                # The final output of the encoder
                # The encoder state is the concatenation of the source encoder and tag encoder
                encoder_state = tf.concat([source_encoder_state, tag_encoder_states], axis=-1)

            # Decode the encoded source and tags with attention
            with tf.variable_scope("decoder"):
                target_embeddings = tf.get_variable("target_embeddings", [target_chars, args.char_dim])

                embedded_target_seqs = tf.nn.embedding_lookup(target_embeddings, target_seqs)
                embedded_target_seqs = tf.layers.dropout(embedded_target_seqs, rate=args.dropout, training=self.is_training)

                # The decoder RNN expects to be the same size as the last dimension of the encoder state,
                # which is the concatenation (sum) of the source encoder and tag encoder states
                decoder_rnn = tf.nn.rnn_cell.GRUCell(args.rnn_dim + args.tag_rnn_dim)

                # Decoder layer used to to classify into target characters.
                decoder_layer = tf.layers.Dense(target_chars)

                # Attention
                source_layer = tf.layers.Dense(args.rnn_dim)
                state_layer = tf.layers.Dense(args.rnn_dim)
                weight_layer = tf.layers.Dense(1)
                source_layer_tag = tf.layers.Dense(args.tag_rnn_dim)
                state_layer_tag = tf.layers.Dense(args.tag_rnn_dim)
                weight_layer_tag = tf.layers.Dense(1)


                def with_attention(inputs, states):
                    """Computes Bahdanau attention on the encoder inputs and states"""

                    # Generate the attention on the source encoder
                    with tf.variable_scope("source_attention"):
                        # Project source_encoded using source_layer.
                        proj_source = source_layer(source_encoder_outputs)

                        # Change shape of states from [a, b] to [a, 1, b] and project it using state_layer.
                        proj_states = state_layer(tf.expand_dims(states, axis=1))

                        # Sum the two above projections, apply tf.tanh and project the result using weight_layer.
                        # The result has shape [x, y, 1].
                        sum_source_states = weight_layer(tf.tanh(proj_source + proj_states))

                        # Apply tf.nn.softmax to the latest result, using axis corresponding to source characters.
                        weight_vec = tf.nn.softmax(sum_source_states, axis=1)

                        # Multiply the source_encoded by the latest result, and sum the results with respect
                        # to the axis corresponding to source characters. This is the final attention.
                        final_attn = tf.reduce_sum(source_encoder_outputs * weight_vec, axis=1)

                    # Generate the attention on the tag encoder
                    with tf.variable_scope("tag_attention"):
                        proj_source_tag = source_layer_tag(tag_encoder_outputs)
                        proj_states_tag = state_layer_tag(tf.expand_dims(states, axis=1))
                        sum_source_states_tag = weight_layer_tag(tf.tanh(proj_source_tag + proj_states_tag))
                        weight_vec_tag = tf.nn.softmax(sum_source_states_tag, axis=1)
                        final_attn_tag = tf.reduce_sum(tag_encoder_outputs * weight_vec_tag, axis=1)

                    # Return concatenation of inputs and the computed attentions.
                    return tf.concat([inputs, final_attn, final_attn_tag], axis=1)

                # The DecoderTraining will be used during training. It will output logits for each
                # target character.
                class DecoderTraining(tf.contrib.seq2seq.Decoder):
                    @property
                    def batch_size(self): return tf.shape(encoder_state)[0]  # Return size of the batch, using encoder states size

                    @property
                    def output_dtype(self): return tf.float32  # Type for logits of target characters

                    @property
                    def output_size(self): return target_chars  # Length of logits for every output

                    def initialize(self, name=None):
                        finished = target_lens <= 0  # False if target_lens > 0, True otherwise
                        states = encoder_state  # Initial decoder state to use
                        inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)),
                                                states)  # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                        return finished, inputs, states

                    def step(self, time, inputs, states, name=None):
                        outputs, states = decoder_rnn(inputs, states)  # Run the decoder GRU cell using inputs and states.
                        outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                        next_input = with_attention(embedded_target_seqs[:, time],
                                                    states)  # Next input is with_attention called on words with index `time` in target_embedded.
                        finished = target_lens <= time + 1  # False if target_lens > time + 1, True otherwise.
                        return outputs, states, next_input, finished

                output_layer, _, _ = tf.contrib.seq2seq.dynamic_decode(DecoderTraining())
                self.predictions_training = tf.argmax(output_layer, axis=2, output_type=tf.int32)

                # The DecoderPrediction will be used during prediction. It will
                # directly output the predicted target characters.
                class DecoderPrediction(tf.contrib.seq2seq.Decoder):
                    @property
                    def batch_size(self): return tf.shape(encoder_state)[0]  # Return size of the batch, using for example source_states size

                    @property
                    def output_dtype(self): return tf.int32  # Type for predicted target characters

                    @property
                    def output_size(self): return 1  # Will return just one output

                    def initialize(self, name=None):
                        finished = tf.fill([self.batch_size], False)  # False of shape [self.batch_size].
                        states = encoder_state  # Initial decoder state to use.
                        inputs = with_attention(tf.nn.embedding_lookup(target_embeddings, tf.fill([self.batch_size], bow)),
                                                states)  # Call with_attention on the embedded BOW characters of shape [self.batch_size].
                        return finished, inputs, states

                    def step(self, time, inputs, states, name=None):
                        outputs, states = decoder_rnn(inputs, states)  # Run the decoder GRU cell using inputs and states.
                        outputs = decoder_layer(outputs)  # Apply the decoder_layer on outputs.
                        outputs = tf.argmax(outputs, output_type=tf.int32,
                                            axis=1)  # Use tf.argmax to choose most probable class (supply parameter `output_type=tf.int32`).
                        next_input = with_attention(tf.nn.embedding_lookup(target_embeddings, outputs),
                                                    states)  # Embed `outputs` using target_embeddings and pass it to with_attention.
                        finished = tf.equal(outputs, eow)  # True where outputs==eow, False otherwise
                        return outputs, states, next_input, finished

                self.predictions, _, self.prediction_lens = tf.contrib.seq2seq.dynamic_decode(
                    DecoderPrediction(), maximum_iterations=tf.reduce_max(source_lens) + 10)

            target_ids = target_seqs

            # Training
            # NOTE: it appears that label smoothing does not help (at least in the range [0.01, 0.1])
            weights = tf.sequence_mask(target_lens, dtype=tf.float32)
            one_hot_labels = tf.one_hot(target_ids, target_chars, axis=2)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels, output_layer, weights=weights,
                                                   label_smoothing=args.label_smoothing)
            global_step = tf.train.get_or_create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate, beta2=args.beta2)

                # Apply gradient clipping
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step, name="training")

            # Summaries
            accuracy_training = tf.reduce_all(tf.logical_or(
                tf.equal(self.predictions_training, target_ids),
                tf.logical_not(tf.sequence_mask(target_lens))), axis=1)
            self.current_accuracy_training, self.update_accuracy_training = tf.metrics.mean(accuracy_training)

            minimum_length = tf.minimum(tf.shape(self.predictions)[1], tf.shape(target_ids)[1])
            accuracy = tf.logical_and(
                tf.equal(self.prediction_lens, target_lens),
                tf.reduce_all(tf.logical_or(
                    tf.equal(self.predictions[:, :minimum_length], target_ids[:, :minimum_length]),
                    tf.logical_not(tf.sequence_mask(target_lens, maxlen=minimum_length))), axis=1))
            self.current_accuracy, self.update_accuracy = tf.metrics.mean(accuracy)

            self.current_loss, self.update_loss = tf.metrics.mean(loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy_training),
                                           tf.contrib.summary.scalar("train/learning_rate", self.learning_rate),]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy),]

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
                sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes  = train.next_batch(batch_size, including_charseqs=True)
                self.session.run(self.reset_metrics)
                self.session.run(
                    [self.training, self.summaries["train"]],
                    {self.sentence_lens: sentence_lens,

                     self.word_ids: word_ids[train.FORMS],

                     self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                     self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                     self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS],

                     self.tags: word_ids[train.TAGS],
                     self.tag_ids: charseq_ids[train.TAGS],
                     self.tag_seqs: charseqs[train.TAGS],
                     self.tag_seq_lens: charseq_lens[train.TAGS],

                     self.learning_rate: learning_rate,
                     self.is_training: True,
                     self.word_indexes: word_indexes})
                pbar.update(len(sentence_lens))

                # if step % 500 == 0:
                #     accuracy = network.evaluate("dev", dev, args.batch_size)
                #
                #     print("{:.2f}".format(100 * accuracy))
                #
                #     if accuracy > best_accuracy:
                #         print('^^ New best ^^')
                #         best_accuracy = accuracy
                #         network.save('model/model')

                step += 1

    def evaluate(self, dataset_name, dataset, batch_size):
        self.session.run(self.reset_metrics)
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes  = dataset.next_batch(batch_size, including_charseqs=True)
            self.session.run([self.update_accuracy, self.update_loss],
            # self.session.run([self.update_accuracy, self.update_loss, self.update_accuracy_tagger, self.update_loss_tagger],
                             {self.sentence_lens: sentence_lens,

                             self.word_ids: word_ids[train.FORMS],

                             self.source_ids: charseq_ids[train.FORMS], self.target_ids: charseq_ids[train.LEMMAS],
                             self.source_seqs: charseqs[train.FORMS], self.target_seqs: charseqs[train.LEMMAS],
                             self.source_seq_lens: charseq_lens[train.FORMS], self.target_seq_lens: charseq_lens[train.LEMMAS],

                             self.tags: word_ids[train.TAGS],
                             self.tag_ids: charseq_ids[train.TAGS],
                             self.tag_seqs: charseqs[train.TAGS],
                             self.tag_seq_lens: charseq_lens[train.TAGS],
                     self.word_indexes: word_indexes})
        return self.session.run([self.current_accuracy, self.summaries[dataset_name]])[0]

    def predict(self, dataset, batch_size):
        lemmas = []
        while not dataset.epoch_finished():
            sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens, word_indexes = dataset.next_batch(batch_size, including_charseqs=True)
            predictions, prediction_lengths = self.session.run(
                [self.predictions, self.prediction_lens],
                {self.sentence_lens: sentence_lens, self.source_ids: charseq_ids[train.FORMS],
                 self.source_seqs: charseqs[train.FORMS], self.source_seq_lens: charseq_lens[train.FORMS],
                 self.word_ids: word_ids[train.FORMS],
                 self.tags: word_ids[train.TAGS],
                 self.tag_ids: charseq_ids[train.TAGS],
                 self.tag_seqs: charseqs[train.TAGS],
                 self.tag_seq_lens: charseq_lens[train.TAGS],
                     self.word_indexes: word_indexes})

            for length in sentence_lens:
                lemmas.append([])
                for i in range(length):
                    lemmas[-1].append("")
                    for j in range(prediction_lengths[i] - 1):
                        lemmas[-1][-1] += train.factors[train.LEMMAS].alphabet[predictions[i][j]]
                predictions, prediction_lengths = predictions[length:], prediction_lengths[length:]

        return lemmas

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--tag_char_dim", default=64, type=int, help="Character embedding dimension for tags.")
    parser.add_argument("--char_dim", default=128, type=int, help="Character embedding dimension.")
    parser.add_argument("--tag_rnn_dim", default=256, type=int, help="Dimension of the encoder and the decoder.")
    parser.add_argument("--rnn_dim", default=350, type=int, help="Dimension of the encoder and the decoder.")
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--label_smoothing", default=0., type=float)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--load", action='store_true')
    parser.add_argument("--train", action='store_true')
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
    dev = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-dev-enrich.txt", train=train, shuffle_batches=False)
    test = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-test-enrich.txt", train=train, shuffle_batches=False)

    analyzer_dictionary = MorphoAnalyzer("czech-pdt/czech-pdt-analysis-dictionary.txt")
    analyzer_guesser = MorphoAnalyzer("czech-pdt/czech-pdt-analysis-guesser.txt")

    # Construct the network
    network = Network()
    network.construct(args, len(train.factors[train.FORMS].alphabet), len(train.factors[train.LEMMAS].alphabet),
                      len(train.factors[train.TAGS].alphabet),
                      len(train.factors[train.TAGS].words), len(train.factors[train.FORMS].words),
                      train.factors[train.LEMMAS].alphabet_map["<bow>"], train.factors[train.LEMMAS].alphabet_map["<eow>"])

    if args.load:
        network.load('model/model')
        accuracy = network.evaluate("dev", dev, args.batch_size)
        print('Final accuracy', accuracy)

    if not args.load or args.train:
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
                network.save('model/model')

                # Predict test data
                with open("lemmatizer_sota_test.txt", "w") as test_file:
                    forms = test.factors[test.FORMS].strings
                    lemmas = network.predict(test, args.batch_size)
                    for s in range(len(forms)):
                        for i in range(len(forms[s])):
                            print("{}\t{}\t_".format(forms[s][i], lemmas[s][i]), file=test_file)
                        print("", file=test_file)

    # Predict test data
    with open("lemmatizer_sota_test.txt", "w") as test_file:
        forms = test.factors[test.FORMS].strings
        lemmas = network.predict(test, args.batch_size)
        for s in range(len(forms)):
            for i in range(len(forms[s])):
                print("{}\t{}\t_".format(forms[s][i], lemmas[s][i]), file=test_file)
            print("", file=test_file)
