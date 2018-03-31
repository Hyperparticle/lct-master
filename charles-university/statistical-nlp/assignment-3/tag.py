import numpy as np
import pandas as pd
import itertools
import nltk
from collections import Counter, defaultdict


def open_text(filename):
    """Reads a text line by line, applies light preprocessing, and returns an array of words and tags"""
    with open(filename, encoding='iso-8859-2') as f:
        content = f.readlines()

    preprocess = lambda word: tuple(word.strip().rsplit('/', 1))

    return [preprocess(word) for word in content]


def isplit(iterable, splitters):
    # https://stackoverflow.com/a/4322780
    return [list(g) for k,g in itertools.groupby(iterable, lambda x:x in splitters) if not k]


def sentence_split(data, token=('###', '###')):
    return isplit(data, (None, token))


def split_data(words, start=0):
    train, heldout, test = words[:start] + words[start+60_000:],  words[start+40_000:start+60_000], words[start:start+40_000]
    return train, heldout, test


def split_data_end(words):
    train, heldout, test = words[:-60_000],  words[-60_000:-40_000], words[-40_000:]
    return train, heldout, test


def split_all(words):
    return [
        split_data_end(words),
        split_data(words, start=40_000 * 0),
        split_data(words, start=40_000 * 1),
        split_data(words, start=40_000 * 2),
        split_data(words, start=40_000 * 3)
    ]


def evaluate(tagger_type, eval_func, langs=('en', 'cz')):
    lang_d = {'en': ('English', splits_en), 'cz': ('Czech', splits_cz)}

    rows = []
    for lang in langs:
        language, splits = lang_d[lang]
        accuracies = [eval_func(split, i, lang) for i, split in enumerate(splits)]
        acc_str = ' '.join(['{0:0.1f}'.format(i * 100) for i in accuracies])
        row = [tagger_type, language, acc_str, np.mean(accuracies) * 100, np.std(accuracies) * 100]
        rows.append(row)

    columns = ['type', 'language', 'accuracies', 'mean', 'standard_deviation']
    results = pd.DataFrame(rows, columns=columns)
    return results


class LISmoother:
    """Linear interpolation smoother"""

    def __init__(self, p_uniform, p_unigram, p_bigram, p_trigram):
        self.p_uniform = p_uniform
        self.p_unigram = p_unigram
        self.p_bigram = p_bigram
        self.p_trigram = p_trigram

        self.lambdas = self.init_lambdas(3)

    def init_lambdas(self, n=3):
        """Initializes a list of lambdas for an ngram language model with uniform probabilities"""
        return np.array([1 / (n + 1)] * (n + 1))

    def smooth(self, heldout_data, stop_tolerance=1e-4):
        """Computes the EM algorithm for linear interpolation smoothing"""

        print('Lambdas:')
        print(self.lambdas)

        next_l = self.next_lambda(self.lambdas, heldout_data)
        while not all(diff < stop_tolerance for diff in np.abs(self.lambdas - next_l)):
            print(next_l)
            self.lambdas = next_l
            next_l = self.next_lambda(self.lambdas, heldout_data)

        print(next_l)
        self.lambdas = next_l

    def next_lambda(self, lambdas, heldout):
        """Computes the next lambda from the current lambdas by normalizing the expected counts"""
        expected = self.expected_counts(lambdas, heldout)
        return expected / np.sum(expected)  # Normalize

    def expected_counts(self, lambdas, heldout):
        """Computes the expected counts by smoothing across all trigrams and summing them all together"""
        smoothed_probs = (self.p_smoothed(lambdas, *h) for h in heldout)  # Multiply lambdas by probabilities
        return np.sum(smoothed / np.sum(smoothed) for smoothed in smoothed_probs)  # Element-wise sum

    def p_smoothed(self, lambdas, tprev, t, w):
        """Calculate the smoothed trigram probability using the weighted product of lambdas"""
        return np.multiply(lambdas, [
            self.p_uniform,
            self.p_unigram[w],
            self.p_bigram[t, w],
            self.p_trigram[tprev, t, w]
        ])


class HMMTagger:
    def __init__(self, tagged_data, tag_set, word_set):
        # Prepend two tokens to avoid beginning-of-data problems

        self.states = tag_set
        self.symbols = word_set

        self.text_size = len(tagged_data)

        # Transition tables - p(t | tprev2, tprev)
        self.transition = defaultdict(float)
        self.transition_bigram = defaultdict(float)
        self.transition_unigram = defaultdict(float)
        self.state_uniform = self.div(1, len(self.states))

        # Emission tables - p(w | tprev, t)
        self.emission = defaultdict(float)
        self.emission_bigram = defaultdict(float)
        self.emission_unigram = defaultdict(float)
        self.symbol_uniform = self.div(1, len(self.symbols))

        unigram_tag_dist = defaultdict(int)
        bigram_tag_dist = defaultdict(int)
        trigram_tag_dist = defaultdict(int)

        unigram_output_dist = defaultdict(int)
        bigram_output_dist = defaultdict(int)
        trigram_output_dist = defaultdict(int)

        tprev, tprev2 = None, None
        for w, t in tagged_data:
            unigram_tag_dist[t] += 1
            bigram_tag_dist[tprev, t] += 1
            trigram_tag_dist[tprev2, tprev, t] += 1

            unigram_output_dist[w] += 1
            bigram_output_dist[t, w] += 1
            trigram_output_dist[tprev, t, w] += 1

            tprev2 = tprev
            tprev = t

        # Build transition tables
        for tprev2, tprev, t in trigram_tag_dist:
            # Use uniform distribution if tags not seen
            if (trigram_tag_dist[tprev2, tprev, t], bigram_tag_dist[tprev, t]) == (0, 0):
                self.transition[tprev2, tprev, t] = self.state_uniform
            self.transition[tprev2, tprev, t] = self.div(trigram_tag_dist[tprev2, tprev, t], bigram_tag_dist[tprev, t])
        #             assert 0 <= self.transition[tprev2, tprev, t] <= 1, self.transition[tprev2, tprev, t]

        for tprev, t in bigram_tag_dist:
            # Use uniform distribution if tags not seen
            if (bigram_tag_dist[tprev, t], unigram_tag_dist[t]) == (0, 0):
                self.transition_bigram[tprev, t] = self.state_uniform
            self.transition_bigram[tprev, t] = self.div(bigram_tag_dist[tprev, t], unigram_tag_dist[t])

        for t in unigram_tag_dist:
            self.transition_unigram[t] = self.div(unigram_tag_dist[t], self.text_size)

        # Build emission tables
        for tprev, t, w in trigram_output_dist:
            # Use uniform distribution if tags not seen
            if (trigram_output_dist[tprev, t, w], bigram_tag_dist[tprev, t]) == (0, 0):
                self.emission[tprev, t, w] = self.symbol_uniform
            self.emission[tprev, t, w] = self.div(trigram_output_dist[tprev, t, w], bigram_tag_dist[tprev, t])

        for t, w in bigram_output_dist:
            # Use uniform distribution if tags not seen
            if (bigram_output_dist[t, w], unigram_tag_dist[t]) == (0, 0):
                self.emission_bigram[t, w] = self.symbol_uniform
            self.emission_bigram[t, w] = self.div(bigram_output_dist[t, w], unigram_tag_dist[t])

        for w in unigram_output_dist:
            self.emission_unigram[w] = self.div(unigram_output_dist[w], self.text_size)

        self.transition_smoother = LISmoother(self.state_uniform, self.transition_unigram, self.transition_bigram,
                                              self.transition)
        self.emission_smoother = LISmoother(self.symbol_uniform, self.emission_unigram, self.emission_bigram,
                                            self.emission)

    def smooth(self, heldout_data):
        """Smooth the transition and emission tables with linear interpolation smoothing"""
        heldout_trigrams = [(tprev, t, w) for (tprev, _), (t, w) in nltk.bigrams(heldout_data)]
        self.transition_smoother.smooth(heldout_trigrams)
        self.emission_smoother.smooth(heldout_trigrams)

    def p_transition(self, tprev2, tprev, t):
        return self.transition_smoother.lambdas.dot([
            self.state_uniform,
            self.transition_unigram[t],
            self.transition_bigram[tprev, t],
            self.transition[tprev2, tprev]
        ])

    def p_emission(self, tprev, t, w):
        return self.emission_smoother.lambdas.dot([
            self.symbol_uniform,
            self.emission_unigram[w],
            self.emission_bigram[t, w],
            self.emission[tprev, t, w]
        ])

    def div(self, a, b):
        """Divides a and b safely"""
        return a / b if b != 0 else 0

    # def p_uniform(self):
    #     """Calculates the probability of choosing a word uniformly at random"""
    #     return self.div(1, self.unigram_count)
    #
    # def p_unigram(self, w):
    #     """Calculates the probability a unigram appears in the distribution"""
    #     return self.div(self.unigram_dist[w], self.total_unigram_count)
    #
    # def p_bigram_cond(self, wprev, w):
    #     """Calculates the probability a word appears in the distribution given the previous word"""
    #     # If neither ngram has been seen, use the uniform distribution for smoothing purposes
    #     if (self.bigram_dist[wprev, w], self.unigram_dist[wprev]) == (0, 0):
    #         return self.p_uniform()
    #
    #     return self.div(self.bigram_dist[wprev, w], self.unigram_dist[wprev])
    #
    # def p_trigram_cond(self, wprev2, wprev, w):
    #     """Calculates the probability a word appears in the distribution given the previous word"""
    #     # If neither ngram has been seen, use the uniform distribution for smoothing purposes
    #     if (self.trigram_dist[wprev2, wprev, w], self.bigram_dist[wprev2, wprev]) == (0, 0):
    #         return self.p_uniform()
    #
    #     return self.div(self.trigram_dist[wprev2, wprev, w], self.bigram_dist[wprev2, wprev])


# Read the texts into memory
english = './data/texten2.ptg'
czech = './data/textcz2.ptg'

words_en = open_text(english)
words_cz = open_text(czech)

splits_en = split_all(words_en)
splits_cz = split_all(words_cz)

train, heldout, test = splits_en[0]

# lambdas = em_algorithm(train, heldout)
#
# print(lambdas)

# words, tags = list(zip(*(train + heldout + test)))
# states, symbols = list(set(tags)), list(set(words))
#
# test = sentence_split(test)
#
# labeled = sentence_split(train[:10_000])
# unlabeled = sentence_split(train[10_000:])
#
# trainer = HMM(states, symbols)
#
# tagger = trainer.train_supervised(labeled)
#
# print('\n')
#
# print(tagger.evaluate(test))
#
# for _ in range(5):
#     tagger = trainer.train_unsupervised(unlabeled, model=tagger, max_iterations=1)
#     print(tagger.evaluate(test))
