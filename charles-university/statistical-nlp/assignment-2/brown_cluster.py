# https://github.com/mheilman/tan-clustering
# https://github.com/percyliang/brown-cluster

import random
import itertools
from collections import defaultdict, Counter, Iterable
from math import isnan, isinf
from tqdm import tqdm
from scipy.special import comb
import pandas as pd
import numpy as np


class LmCluster(object):
    def __init__(self, tokens, word_cutoff=10, cluster_cutoff=1):
        self.tokens = tokens
        self.word_cutoff = word_cutoff
        self.cluster_cutoff = cluster_cutoff

        self.num_tokens = len(self.tokens)
        self.word_counts, self.bigram_counts, self.word2int, self.int2word = self.freq_dist(self.tokens)

        # Hierarchical mapping of cluster IDs
        self.cluster_parents = {}
        self.cluster_history = []
        # self.cluster_bits = {}  # the bit to add when walking up the hierarchy from a word to the top-level cluster
        self.cluster_counter = len(self.word2int)

        self.classes = [word for word in range(len(self.word2int)) if self.word_counts[word] >= self.word_cutoff]

        # classes = [c for c in self.classes if self.counts[c] >= self.word_cutoff]
        # classes = self.classes

        # The graph weights W and the results of merging nodes L (Liang's thesis)
        self.W = self.build_w(self.classes)
        self.L = self.build_l(self.classes)

        print('Word tokens: {}'.format(self.num_tokens))
        print('Starting classes: {}'.format(len(self.classes)))
        print('Starting MI:', sum(self.W[c1][c2] for c1 in self.W for c2 in self.W[c1]))

        scores = sorted([(*self.class_name([c1, c2]), score) for c1 in self.L for c2, score in self.L[c1].items()],
                        key=lambda s: s[2], reverse=True)[:20]

        print('\n'.join([str(s) for s in scores]))

        merges = len(self.classes) - self.cluster_cutoff
        for _ in tqdm(range(merges), unit='class'):
            # Merge the classes that reduce the mutual information the least
            c1, c2 = self.find_best_merge()
            c_new = self.merge_classes(c1, c2)

            # Add classes to the history of merges
            self.cluster_history.append((*self.class_name([c1, c2]), c_new))

            self.cluster_counter += 1

    def class_name(self, classes):
        if not isinstance(classes, Iterable):
            classes = [classes]

        classes = [self.int2word[c] if c < len(self.int2word) else str(c) for c in classes]
        return classes if len(classes) > 1 else classes[0]

    @staticmethod
    def freq_dist(tokens):
        counts = Counter(tokens)
        # word_set = sorted(counts.keys(), key=lambda word: counts[word], reverse=True)
        word_set = counts.keys()

        word2int = {}
        word_counts = defaultdict(int)

        for i, w in enumerate(word_set):
            word2int[w] = i
            word_counts[word2int[w]] = counts[w]

        int2word = sorted(word2int.keys(), key=lambda word: word2int[word])

        bigram_counts = defaultdict(lambda: defaultdict(int))
        for w1, w2 in zip(tokens, tokens[1:]):
            bigram_counts[word2int[w1]][word2int[w2]] += 1

        return word_counts, bigram_counts, word2int, int2word

    def build_w(self, classes):
        W = defaultdict(lambda: defaultdict(float))

        # edges between nodes
        for c1, c2 in itertools.combinations(classes, 2):
            W[c1][c2] = self.compute_weight([c1], [c2]) + self.compute_weight([c2], [c1])

        # edges to and from a single node
        for c in classes:
            W[c][c] = self.compute_weight([c], [c])

        return W

    def build_l(self, classes):
        L = defaultdict(lambda: defaultdict(float))

        total = comb(len(classes), 2, exact=True)
        for c1, c2 in tqdm(itertools.combinations(classes, 2), total=total, unit='pairs'):
            L[c1][c2] = self.compute_l(c1, c2)

        return L

    def compute_weight(self, nodes1, nodes2):
        paircount = sum(self.bigram_counts[n1][n2] for n1 in nodes1 for n2 in nodes2)

        if not paircount:
            return 0.0

        count_1 = sum(self.word_counts[n] for n in nodes1)
        count_2 = sum(self.word_counts[n] for n in nodes2)

        return (paircount / self.num_tokens) * np.log2(paircount * self.num_tokens / count_1 / count_2)

    def compute_l(self, c1, c2):
        val = 0.0

        # classes = classes = [c for c in self.classes if self.counts[c] >= self.word_cutoff]
        classes = self.classes

        # add the weight of edges coming in to the potential
        # new cluster from other nodes
        # TODO this is slow
        for d in classes:
            val += self.compute_weight([c1, c2], [d])
            val += self.compute_weight([d], [c1, c2])

        # ... but don't include what will be part of the new cluster
        for d in [c1, c2]:
            val -= self.compute_weight([c1, c2], [d])
            val -= self.compute_weight([d], [c1, c2])

        # add the weight of the edge from the potential new cluster
        # to itself
        val += self.compute_weight([c1, c2], [c1, c2])

        # subtract the weight of edges to/from c1, c2
        # (which would be removed)
        for d in classes:
            for c in [c1, c2]:
                if d in self.W[c]:
                    val -= self.W[c][d]
                elif c in self.W[d]:
                    val -= self.W[d][c]

        return val

    def find_best_merge(self):
        best_score = float('-inf')
        argmax = None

        for c1 in self.L:
            for c2, score in self.L[c1].items():
                if score > best_score:
                    argmax = [(c1, c2)]
                    best_score = score
                elif score == best_score:
                    argmax.append((c1, c2))

        if isnan(best_score) or isinf(best_score):
            raise ValueError("bad value for score: {}".format(best_score))

        # break ties randomly (randint takes inclusive args!)
        #         c1, c2 = argmax[random.randint(0, len(argmax) - 1)]
        c1, c2 = argmax[0]

        # print('{0: <10} + {1: <10} -> {2: <10}'.format(*self.class_name([c1, c2]), self.cluster_counter))

        return c1, c2

    def merge_classes(self, c1, c2):
        c_new = self.cluster_counter

        # record parents
        self.cluster_parents[c1] = c_new
        self.cluster_parents[c2] = c_new
        # r = random.randint(0, 1)
        # self.cluster_bits[c1] = str(r)  # assign bits randomly
        # self.cluster_bits[c2] = str(1 - r)

        # add the new cluster to the counts and transitions dictionaries
        self.word_counts[c_new] = self.word_counts[c1] + self.word_counts[c2]
        for c in [c1, c2]:
            for d, val in self.bigram_counts[c].items():
                if d == c1 or d == c2:
                    d = c_new
                self.bigram_counts[c_new][d] += val

        # subtract the weights for the merged nodes from the score table
        # TODO this is slow
        for c in [c1, c2]:
            for d1 in self.L:
                for d2 in self.L[d1]:
                    self.L[d1][d2] -= self.compute_weight([d1, d2], [c])
                    self.L[d1][d2] -= self.compute_weight([c], [d1, d2])

        # remove merged clusters from the counts and transitions dictionaries
        # to save memory (but keep frequencies for words for the final output)
        if c1 >= len(self.word2int):
            del self.word_counts[c1]
        if c2 >= len(self.word2int) and c2 in self.word_counts:
            del self.word_counts[c2]

        del self.bigram_counts[c1]
        del self.bigram_counts[c2]
        for d in self.bigram_counts:
            for c in [c1, c2]:
                if c in self.bigram_counts[d]:
                    del self.bigram_counts[d][c]

        # remove the old clusters from the w and L tables
        for table in [self.W, self.L]:
            for d in table:
                if c1 in table[d]:
                    del table[d][c1]
                if c2 in table[d]:
                    del table[d][c2]
            if c1 in table:
                del table[c1]
            if c2 in table:
                del table[c2]

        # remove the merged items
        self.classes.remove(c1)
        self.classes.remove(c2)

        # add the new cluster to the w and L tables
        self.add_to_batch(c_new)

        return c_new

    def add_to_batch(self, c_new):
        # compute weights for edges connected to the new node
        for d in self.classes:
            self.W[d][c_new] = self.compute_weight([d], [c_new])
            self.W[d][c_new] = self.compute_weight([c_new], [d])
        self.W[c_new][c_new] = self.compute_weight([c_new], [c_new])

        # add the weights from this new node to the merge score table
        # TODO this is slow
        for d1 in self.L:
            for d2 in self.L[d1]:
                self.L[d1][d2] += self.compute_weight([d1, d2], [c_new])
                self.L[d1][d2] += self.compute_weight([c_new], [d1, d2])

        # compute scores for merging it with all clusters in the current batch
        for d in self.classes:
            self.compute_l(d, c_new)

        # now add it to the batch
        self.classes.append(c_new)

    # def get_bitstring(self, w):
    #     # walk up the cluster hierarchy until there is no parent cluster
    #     cur_cluster = self.word2int[w]
    #     bitstring = ""
    #     while cur_cluster in self.cluster_parents:
    #         bitstring = self.cluster_bits[cur_cluster] + bitstring
    #         cur_cluster = self.cluster_parents[cur_cluster]
    #     return bitstring
    #
    # def save_clusters(self, output_path):
    #     with open(output_path, 'w') as f:
    #         for w in self.word2int:
    #             # convert the counts back to ints when printing
    #             f.write("{}\t{}\t{}\n".format(w, self.get_bitstring(w), self.word_counts[self.word2int[w]]))
    #
    # def print_clusters(self):
    #     for w in self.word2int:
    #         print("{}\t{}\t{}".format(w, self.get_bitstring(w), self.word_counts[self.word2int[w]]))


def preprocess(word):
    return word.strip()


def open_text(filename):
    """Reads a text line by line, applies light preprocessing, and returns an array of words"""
    with open(filename, encoding='iso-8859-2') as f:
        content = f.readlines()

    return [preprocess(word) for word in content]


def history(cluster):
    return pd.DataFrame(cluster.cluster_history, columns=['prev word', 'word', 'cluster id'])


if __name__ == '__main__':
    random.seed(100)

    english = './TEXTEN1.txt'
    words_en = open_text(english)

    lm_cluster = LmCluster(words_en[:8000])

    print(history(lm_cluster)[:5])
