import numpy as np
import pandas as pd
import itertools
import nltk
from tqdm import tqdm_notebook as tqdm
from collections import defaultdict


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
    return [[(token[0], token[0])] + g for g in isplit(data, (None, token))]


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


def neg_inf_array(shape):
    res = np.empty(shape, np.float64)
    res.fill(-np.inf)
    return res


def logsumexp2(arr):
    max_ = arr.max()
    return np.log2(np.sum(2 ** (arr - max_))) + max_


def log_add(*values):
    x = max(values)
    if x > -np.inf:
        sum_diffs = 0
        for value in values:
            sum_diffs += 2 ** (value - x)
        return x + np.log2(sum_diffs)
    else:
        return x


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

        # print('Lambdas:')
        # print(self.lambdas)

        next_l = self.next_lambda(self.lambdas, heldout_data)
        while not all(diff < stop_tolerance for diff in np.abs(self.lambdas - next_l)):
            # print(next_l)
            self.lambdas = next_l
            next_l = self.next_lambda(self.lambdas, heldout_data)

        # print(next_l)
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
            # self.p_trigram[tprev, t, w]
        ])


class HMMTagger:
    def __init__(self, tagged_data, tag_set, word_set):
        # Prepend two tokens to avoid beginning-of-data problems

        self.states = list(sorted(tag_set))
        self.symbols = list(sorted(word_set))

        self.istates = np.arange(0, len(self.states))
        self.states2i = {s: i for i, s in enumerate(self.states)}

        self.isymbols = np.arange(0, len(self.symbols))
        self.symbols2i = {s: i for i, s in enumerate(self.symbols)}

        self.text_size = sum(len(sequence) for sequence in tagged_data)

        # Transition tables - p(t | tprev)
        self.transition_trigram = defaultdict(float)
        self.transition_bigram = defaultdict(float)
        self.transition_unigram = defaultdict(float)
        self.state_uniform = self.div(1, len(self.states))

        # Emission tables - p(w | tprev)
        self.emission_bigram = defaultdict(float)
        self.emission_unigram = defaultdict(float)
        self.symbol_uniform = self.div(1, len(self.symbols))

        self.prior_probs = defaultdict(float)

        self.train_supervised(tagged_data)

    def train_supervised(self, tagged_data):
        unigram_tag_dist = defaultdict(int)
        bigram_tag_dist = defaultdict(int)
        trigram_tag_dist = defaultdict(int)

        unigram_output_dist = defaultdict(int)
        bigram_output_dist = defaultdict(int)
        trigram_output_dist = defaultdict(int)

        prior_dist = defaultdict(int)

        for sequence in tagged_data:
            tprev, tprev2 = None, None
            for w, t in sequence:
                # if tprev is None:
                #     prior_dist[t] += 1
                # else:
                unigram_tag_dist[t] += 1
                bigram_tag_dist[tprev, t] += 1
                trigram_tag_dist[tprev2, tprev, t] += 1

                unigram_output_dist[w] += 1
                bigram_output_dist[t, w] += 1
                trigram_output_dist[tprev, t, w] += 1

                tprev2 = tprev
                tprev = t

        # Prior distribution
        for t in self.states:
            if prior_dist[t] == 0:
                self.prior_probs[t] = self.state_uniform
            else:
                self.prior_probs[t] = self.div(prior_dist[t], len(self.states))

        # Build transition tables
        for tprev2, tprev, t in trigram_tag_dist:
            # Use uniform distribution if tags not seen
            if (trigram_tag_dist[tprev2, tprev, t], bigram_tag_dist[tprev, t]) == (0, 0):
                self.transition_trigram[tprev2, tprev, t] = self.state_uniform
            else:
                self.transition_trigram[tprev2, tprev, t] = self.div(trigram_tag_dist[tprev2, tprev, t], bigram_tag_dist[tprev, t])

        for tprev, t in bigram_tag_dist:
            if (bigram_tag_dist[tprev, t], unigram_tag_dist[t]) == (0, 0):
                self.transition_bigram[tprev, t] = self.state_uniform
            else:
                self.transition_bigram[tprev, t] = self.div(bigram_tag_dist[tprev, t], unigram_tag_dist[t])

        for t in unigram_tag_dist:
            self.transition_unigram[t] = self.div(unigram_tag_dist[t], self.text_size)

        # Build emission tables
        for t, w in bigram_output_dist:
            if (bigram_output_dist[t, w], unigram_tag_dist[t]) == (0, 0):
                self.emission_bigram[t, w] = self.symbol_uniform
            else:
                self.emission_bigram[t, w] = self.div(bigram_output_dist[t, w], unigram_tag_dist[t])

        for w in unigram_output_dist:
            self.emission_unigram[w] = self.div(unigram_output_dist[w], self.text_size)

        self.transition_smoother = LISmoother(self.state_uniform, self.transition_unigram,
                                              self.transition_bigram, self.transition_trigram)
        self.emission_smoother = LISmoother(self.symbol_uniform, self.emission_unigram,
                                            self.emission_bigram)

        num_states = len(self.states)
        num_symbols = len(self.symbols)

        self.transitions = neg_inf_array((num_states, num_states))
        self.emissions = neg_inf_array((num_states, num_symbols))

        self.priors = np.zeros(len(self.states), np.float32)
        for i in range(num_states):
            self.priors[i] = np.log2(self.prior_probs[self.states[i]])

        # for i in range(num_states):
        #     for j in range(num_states):
        #         self.transitions[i, j] = np.log2(self.p_transition(self.states[i], self.states[j]))
        #     for k in range(num_symbols):
        #         self.emissions[i, k] = np.log2(self.p_emission(self.states[i], self.symbols[k]))

    def smooth(self, heldout_data):
        """Smooth the transition and emission tables with linear interpolation smoothing"""
        heldout_trigrams = [(tprev, t, w) for (tprev, _), (t, w) in nltk.bigrams(heldout_data)]
        # print("Smoothing transition table")
        self.transition_smoother.smooth(heldout_trigrams)
        # print()

        # print("Smoothing emission table")
        self.emission_smoother.smooth(heldout_trigrams)
        # print()

        num_states = len(self.states)
        num_symbols = len(self.symbols)

        for i in range(num_states):
            for j in range(num_states):
                self.transitions[i, j] = np.log2(self.p_transition(self.states[i], self.states[j]))
            for k in range(num_symbols):
                self.emissions[i, k] = np.log2(self.p_emission(self.states[i], self.symbols[k]))

    def train_unsupervised(self, unlabeled_sequences, max_iterations=5, update_outputs=True):
        N = len(self.states)
        M = len(self.symbols)

        converged = False
        last_logprob = float('-inf')
        iteration = 0
        epsilon = 1e-6

        while not converged and iteration < max_iterations:
            A_numer = neg_inf_array((N, N))
            B_numer = neg_inf_array((N, M))
            A_denom = neg_inf_array(N)
            B_denom = neg_inf_array(N)

            logprob = 0
            for sequence in tqdm(unlabeled_sequences):
                sequence = list(sequence)
                if not sequence:
                    continue

                (lpk, seq_A_numer, seq_A_denom, seq_B_numer, seq_B_denom) = self.forward_backward(sequence)

                # add these sums to the global A and B values
                for i in range(N):
                    A_numer[i] = np.logaddexp2(A_numer[i], seq_A_numer[i] - lpk)
                    B_numer[i] = np.logaddexp2(B_numer[i], seq_B_numer[i] - lpk)

                A_denom = np.logaddexp2(A_denom, seq_A_denom - lpk)
                B_denom = np.logaddexp2(B_denom, seq_B_denom - lpk)

                logprob += lpk

            # use the calculated values to update the transition and output
            # probability values
            for i in range(N):
                logprob_Ai = A_numer[i] - A_denom[i]
                logprob_Bi = B_numer[i] - B_denom[i]

                # We should normalize all probabilities (see p.391 Huang et al)
                # Let sum(P) be K.
                # We can divide each Pi by K to make sum(P) == 1.
                #   Pi' = Pi/K
                #   log2(Pi') = log2(Pi) - log2(K)
                logprob_Ai -= logsumexp2(logprob_Ai)
                logprob_Bi -= logsumexp2(logprob_Bi)

                # update output and transition probabilities
                si = self.istates[i]

                for j in range(N):
                    sj = self.istates[j]
                    self.transitions[si, sj] = logprob_Ai[j]

                if update_outputs:
                    for k in range(M):
                        ok = self.isymbols[k]
                        self.emissions[si, ok] = logprob_Bi[k]

            # test for convergence
            if iteration > 0 and abs(logprob - last_logprob) < epsilon:
                converged = True

            print('iteration', iteration, 'logprob', logprob)
            iteration += 1
            last_logprob = logprob

    def forward_backward(self, sequence):
        N = len(self.states)
        M = len(self.symbols)
        T = len(sequence)

        # compute forward and backward probabilities
        alpha = self.forward(sequence)
        beta = self.backward(sequence)

        # find the log probability of the sequence
        lpk = logsumexp2(alpha[T-1])

        A_numer = neg_inf_array((N, N))
        B_numer = neg_inf_array((N, M))
        A_denom = neg_inf_array(N)
        B_denom = neg_inf_array(N)

        transitions_logprob = self.transitions.T

        for t in range(T):
        # for t in range(T - 1):
            symbol = sequence[t]
            next_symbol = '###'
            if t < T - 1:
                next_symbol = sequence[t+1]
            xi = self.symbols2i[symbol]

            next_outputs_logprob = self.emissions[:, self.symbols2i[next_symbol]]
            alpha_plus_beta = alpha[t] + beta[t]

            if t < T - 1:
                numer_add = transitions_logprob + next_outputs_logprob + \
                            beta[t+1] + alpha[t].reshape(N, 1)
                A_numer = np.logaddexp2(A_numer, numer_add)
                A_denom = np.logaddexp2(A_denom, alpha_plus_beta)
            else:
                B_denom = np.logaddexp2(A_denom, alpha_plus_beta)

            B_numer[:,xi] = np.logaddexp2(B_numer[:,xi], alpha_plus_beta)

        return lpk, A_numer, A_denom, B_numer, B_denom

    def forward(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self.states)
        alpha = neg_inf_array((T, N))

        transitions_logprob = self.transitions

        # Initialization
        symbol = unlabeled_sequence[0]
        for i, state in enumerate(self.istates):
            alpha[0, i] = self.emissions[state, self.symbols2i[symbol]]

        # Induction
        for t in range(1, T):
            symbol = unlabeled_sequence[t]
            output_logprob = self.emissions[:, self.symbols2i[symbol]]

            for i in range(N):
                summand = alpha[t - 1] + transitions_logprob[i]
                alpha[t, i] = logsumexp2(summand) + output_logprob[i]

        return alpha

    def backward(self, unlabeled_sequence):
        T = len(unlabeled_sequence)
        N = len(self.states)
        beta = neg_inf_array((T, N))

        transitions_logprob = self.transitions.T

        # initialise the backward values;
        # "1" is an arbitrarily chosen value from Rabiner tutorial
        beta[T - 1, :] = np.log2(1)

        # inductively calculate remaining backward values
        for t in range(T - 2, -1, -1):
            symbol = unlabeled_sequence[t + 1]
            outputs = self.emissions[:, self.symbols2i[symbol]]

            for i in range(N):
                summand = transitions_logprob[i] + beta[t + 1] + outputs
                beta[t, i] = logsumexp2(summand)

        return beta

    def tag(self, words):
        seqlen = len(words)
        num_states = len(self.states)

        seq_probs = np.zeros((seqlen, num_states), np.float32)
        backpointers = -np.ones((seqlen, num_states), np.int)

        # Find the starting probabilities for each state
        seq_probs[0] = self.priors + self.emissions[:, self.symbols2i[words[0]]]

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

    def div(self, a, b):
        """Divides a and b safely"""
        return a / b if b != 0 else 0


if __name__ == "__main__":
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

    labeled_train = sentence_split(train[:10_000])
    # labeled_train = sentence_split(train)
    unlabeled_train = [list(zip(*sentence))[0] for sentence in sentence_split(train[10_000:])]
    test_sentences = sentence_split(test)

    tagger = HMMTagger(labeled_train, tag_set, word_set)
    tagger.smooth(heldout)

    tagger.train_unsupervised(unlabeled_train, max_iterations=1)

    print(tagger.evaluate(test_sentences))
    print()

    # unlabeled_words = [list(zip(*sentence))[0] for sentence in sentence_split(unlabeled)]
    # tagger.train_unsupervised(unlabeled_words, max_iterations=20)
    #
    # print(tagger.evaluate(sentences))
    # print()
