import numpy as np
import pandas as pd
import itertools
import nltk
from tqdm import tqdm
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

    def __init__(self, p_uniform, p_unigram, p_bigram, p_trigram=None):
        self.p_uniform = p_uniform
        self.p_unigram = p_unigram
        self.p_bigram = p_bigram
        self.p_trigram = p_trigram

        self.lambdas = self.init_lambdas(2)
#         self.lambdas = self.init_lambdas(3)

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
            #             self.p_trigram[tprev, t, w]
        ])


class HMMTagger:
    def __init__(self, tagged_data, tag_set, word_set):
        # Prepend two tokens to avoid beginning-of-data problems

        self.states = list(sorted(tag_set))
        self.symbols = list(sorted(word_set))

        self.text_size = len(tagged_data)

        # Transition tables - p(t | tprev2, tprev)
        self.transition_trigram = defaultdict(float)
        self.transition_bigram = defaultdict(float)
        self.transition_unigram = defaultdict(float)
        self.state_uniform = self.div(1, len(self.states))

        # Emission tables - p(w | tprev, t)
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
                self.transition_trigram[tprev2, tprev, t] = self.state_uniform
            self.transition_trigram[tprev2, tprev, t] = self.div(trigram_tag_dist[tprev2, tprev, t], bigram_tag_dist[tprev, t])

        for tprev, t in bigram_tag_dist:
            # Use uniform distribution if tags not seen
            if (bigram_tag_dist[tprev, t], unigram_tag_dist[t]) == (0, 0):
                self.transition_bigram[tprev, t] = self.state_uniform
            self.transition_bigram[tprev, t] = self.div(bigram_tag_dist[tprev, t], unigram_tag_dist[t])

        for t in unigram_tag_dist:
            self.transition_unigram[t] = self.div(unigram_tag_dist[t], self.text_size)

        # Build emission tables
        for t, w in bigram_output_dist:
            # Use uniform distribution if tags not seen
            if (bigram_output_dist[t, w], unigram_tag_dist[t]) == (0, 0):
                self.emission_bigram[t, w] = self.symbol_uniform
            self.emission_bigram[t, w] = self.div(bigram_output_dist[t, w], unigram_tag_dist[t])

        for w in unigram_output_dist:
            self.emission_unigram[w] = self.div(unigram_output_dist[w], self.text_size)

        self.transition_smoother = LISmoother(self.state_uniform, self.transition_unigram,
                                              self.transition_bigram, self.transition_trigram)
        self.emission_smoother = LISmoother(self.symbol_uniform, self.emission_unigram,
                                            self.emission_bigram)

        self.istates = np.arange(0, len(self.states))
        self.states2i = {s: i for i, s in enumerate(self.states)}

        self.isymbols = np.arange(0, len(self.symbols))
        self.symbols2i = {s: i for i, s in enumerate(self.symbols)}

        num_states = len(self.states)
        num_symbols = len(self.symbols)

        self.transitions = np.zeros((num_states, num_states), np.float32)
        self.emissions = np.zeros((num_states, num_symbols), np.float32)

    def smooth(self, heldout_data):
        """Smooth the transition and emission tables with linear interpolation smoothing"""
        heldout_trigrams = [(tprev, t, w) for (tprev, _), (t, w) in nltk.bigrams(heldout_data)]
        print("Smoothing transition table")
        self.transition_smoother.smooth(heldout_trigrams)

        print("Smoothing emission table")
        self.emission_smoother.smooth(heldout_trigrams)
        print()

        num_states = len(self.states)
        num_symbols = len(self.symbols)

        for i in range(num_states):
            for j in range(num_states):
                self.transitions[i, j] = np.log2(self.p_transition(self.states[i], self.states[j]))
            for k in range(num_symbols):
                self.emissions[i, k] = np.log2(self.p_emission(self.states[i], self.symbols[k]))

    # def train_unsupervised(self, unlabeled_sequences, max_iterations=50):
    #     converged = False
    #     iteration = 1
    #
    #     while not converged and iteration < max_iterations:
    #         for sequence in unlabeled_sequences:
    #             n = len(sequence)
    #             m = len(self.states)
    #             Yt = [self.symbols.index(yt) for yt in sequence]
    #
    #             alpha, beta, gamma = self.forward_backward(sequence)
    #
    #             xsi = np.zeros((n, m, m), dtype=np.float)
    #             for t in range(n - 1):
    #                 for i in range(m):
    #                     for j in range(m):
    #                         xsi[t, i, j] = (
    #                                 alpha[t, i] *
    #                                 self.p_transition(i, j) *
    #                                 beta[t + 1, j] *
    #                                 self.p_emission(j, Yt[t + 1])
    #                         )
    #                 xsi[t] /= np.sum(xsi[t])
    #
    #             # Update
    #             pi = gamma[0]
    #
    #             transition_prob = np.zeros((m, m), dtype=np.float)
    #             for i in range(m):
    #                 den = np.sum(gamma[:, i])
    #                 for j in range(m):
    #                     transition_prob[i, j] = np.sum(xsi[:, i, j]) / den
    #
    #             emission_prob = np.zeros((m, len(self.symbols)), dtype=np.float)
    #             for i in range(m):
    #                 den = np.sum(gamma[:, i])
    #                 for j in range(len(self.symbols)):
    #                     emission_prob[i, j] = np.sum(gamma[np.array(Yt) == j, i]) / den
    #
    #         diff_transition = np.max(hmm.transition_prob - transition_prob)
    #         diff_emission = np.max(hmm.emission_prob - emission_prob)
    #         converged = diff_transition < eps and diff_emission < eps
    #
    #         iteration += 1
    #
    # def forward_backward(self, sequence):
    #     Yt = [Y.index(yt) for yt in sequence]
    #     n = len(sequence)
    #     m = len(X)
    #
    #     alpha = self.forward(Yt)
    #     beta = self.backward(Yt)
    #     gamma = np.zeros((n, m), dtype=np.float)
    #     for t in range(n):
    #         gamma[t, :] = [alpha[t, i] * beta[t, i] for i in range(m)]
    #         gamma[t, :] /= np.sum(gamma[t, :])
    #
    #     return alpha, beta, gamma
    #
    # def forward(self, Yt):
    #     n = len(Yt)
    #     m = len(X)
    #     alpha = np.zeros((n, m), dtype=np.float)
    #
    #     alpha[0, :] = (
    #         hmm.initial_prob
    #             .dot(np.diag(self.p_emission([:, Yt[0]])))
    #     )
    #
    #     for t in xrange(1, n):
    #         alpha[t, :] = (
    #             alpha[t - 1, :]
    #                 .dot(hmm.transition_prob)
    #                 .dot(np.diag(hmm.emission_prob[:, Yt[t]]))
    #         )
    #     return alpha
    #
    # def backward(self, Yt):
    #     n = len(Yt)
    #     m = len(X)
    #     beta = np.zeros((n, m), dtype=mpf)
    #
    #     beta[n - 1, :] = [1.0] * m
    #
    #     for t in range(n - 2, -1, -1):
    #         beta[t, :] = (
    #             self.p_transition
    #                 .dot(np.diag(hmm.emission_prob[:, Yt[t + 1]]))
    #                 .dot(beta[t + 1, :].T)
    #         )
    #     return beta

    def tag(self, words):
        # prev_states = [x for x in self.states if x[0] == None] if t == 1 else self.states
        # best_states = list(sorted(((s, V[t - 1, s]) for s in tagger.states), key=lambda x: x[1], reverse=True))[:n_best]
        # best_states = [x[0] for x in best_states]
        seqlen = len(words)
        num_states = len(self.states)

        seq_probs = np.zeros((seqlen, num_states), np.float32)
        backpointers = -np.ones((seqlen, num_states), np.int)

        # Find the starting probabilities for each state
        seq_probs[0] = self.emissions[:, self.symbols2i[words[0]]]

        # Find the maximum probabilities for reaching each state at time t
        for t in range(1, seqlen):
            for j in range(num_states):
                vs = seq_probs[t-1, :] + self.transitions[:, j]
                best = np.argmax(vs)
                seq_probs[t, j] = vs[best] + self.emissions[j, self.symbols2i[words[t]]]
                backpointers[t, j] = best

        current = np.argmax(seq_probs[seqlen - 1, :])
        sequence = [current]
        for t in range(seqlen - 1, 0, -1):
            last = backpointers[t, current]
            sequence.append(last)
            current = last

        sequence.reverse()
        return [self.states[s] for s in sequence]

    def evaluate(self, data):
        total, correct = 0, 0
        for sentence in tqdm(data):
            words, tags = zip(*sentence)
            predicted_tags = self.tag(words)
            for tag, pred in zip(tags, predicted_tags):
                if tag == pred:
                    correct += 1
                total += 1

        return correct / total

    def p_transition(self, tprev, t):
        return self.transition_smoother.lambdas.dot([
            self.state_uniform,
            self.transition_unigram[t],
            self.transition_bigram[tprev, t]
        ])

    def p_emission(self, t, w):
        return self.emission_smoother.lambdas.dot([
            self.symbol_uniform,
            self.emission_unigram[w],
            self.emission_bigram[t, w]
        ])

    #     def p_transition(self, tprev, t):
    #         tprev2, tprev = tprev
    #         return self.transition_smoother.lambdas.dot([
    #             self.state_uniform,
    #             self.transition_unigram[t],
    #             self.transition_bigram[tprev, t],
    #             self.transition[tprev2, tprev]
    #         ])

    #     def p_emission(self, t, w):
    #         tprev, t = t
    #         return self.emission_smoother.lambdas[:3].dot([
    #             self.symbol_uniform,
    #             self.emission_unigram[w],
    #             self.emission_bigram[t, w]
    #         ])

    def div(self, a, b):
        """Divides a and b safely"""
        return a / b if b != 0 else 0


# Read the texts into memory
english = './data/texten2.ptg'
czech = './data/textcz2.ptg'

words_en = open_text(english)
words_cz = open_text(czech)

splits_en = split_all(words_en)
splits_cz = split_all(words_cz)

train, heldout, test = splits_en[0]

words, tags = list(zip(*(train + heldout + test)))
tag_set, word_set = list(set(tags)), list(set(words))
# tag_set, word_set = set(nltk.bigrams(tags, pad_left=True)), set(words)

labeled = train[:10_000]
unlabeled = train[10_000:]

tagger = HMMTagger(labeled, tag_set, word_set)
tagger.smooth(heldout)

sentences = sentence_split(test)

words, tags = list(zip(*sentences[0]))

print("Tagger:")
print(tagger.tag(words))

print(tags)
print(words)

# print()
# print(tagger.evaluate(sentences[:20]))
