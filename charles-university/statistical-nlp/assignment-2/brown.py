# https://github.com/mheilman/tan-clustering
# https://github.com/percyliang/brown-cluster
import random
import argparse
import glob
import re
import itertools
import logging
from collections import defaultdict
from math import log, isnan, isinf
from tqdm import tqdm
from scipy.special import comb

random.seed(100)

logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(message)s')

def open_text(filename):
    """Reads a text line by line, applies light preprocessing, and returns an array of words"""
    with open(filename, encoding='iso-8859-2') as f:
        content = f.readlines()
    
    preprocess = lambda word: word.strip()
    
    return [preprocess(word) for word in content]

class ClassLMClusters(object):
    def __init__(self, tokens, batch_size=1000, max_vocab_size=None,
                 lower=False, word_cutoff=10, cluster_cutoff=0):

        self.batch_size = batch_size
        self.tokens = tokens
        self.lower = lower  # whether to lowercase everything

        self.max_vocab_size = max_vocab_size
        self.word_cutoff = word_cutoff
        self.cluster_cutoff = cluster_cutoff

        # mapping from cluster IDs to cluster IDs,
        # to keep track of the hierarchy
        self.cluster_parents = {}
        self.cluster_counter = 0

        # the list of words in the vocabulary and their counts
        self.counts = defaultdict(int)
        self.trans = defaultdict(lambda: defaultdict(int))
        self.num_tokens = 0

        # the graph weights (w) and the effects of merging nodes (L)
        # (see Liang's thesis)
        self.w = defaultdict(lambda: defaultdict(float))
        self.L = defaultdict(lambda: defaultdict(float))

        # the 0/1 bit to add when walking up the hierarchy
        # from a word to the top-level cluster
        self.cluster_bits = {}

        # find the most frequent words
        self.vocab = {}
        self.reverse_vocab = []
        self.create_vocab()

        # create sets of documents that each word appears in
        self.create_index()

        # make a copy of the list of words, as a queue for making new clusters
        word_queue = list(range(len(self.vocab)))

        # score potential clusters, starting with the most frequent words.
        # also, remove the batch from the queue
#         self.current_batch = word_queue[:(self.batch_size + 1)]
#         word_queue = word_queue[(self.batch_size + 1):]
        
#         self.current_batch = word_queue
#         word_queue = []
        
        self.current_batch = [word for word in word_queue if self.counts[word] >= word_cutoff]
        print(len(word_queue), len(self.current_batch))
        word_queue = []
        
        self.initialize_tables()

        with tqdm(total=len(self.current_batch) + len(word_queue) - 1, unit='class') as t:
            while len(self.current_batch) > 1:
                # find the best pair of words/clusters to merge
                c1, c2 = self.find_best()

                # merge the clusters in the index
                self.merge(c1, c2)

                if word_queue:
                    new_word = word_queue.pop(0)
                    self.add_to_batch(new_word)

#                 t.update(1)
                self.cluster_counter += 1

                logging.info('{0: <10} + {1: <10} -> {2: <10}'
                             .format(self.reverse_vocab[c1] if c1 < len(self.reverse_vocab) else c1,
                                     self.reverse_vocab[c2] if c2 < len(self.reverse_vocab) else c2,
                                     self.cluster_counter))
                
#                 logging.info('{} AND {} WERE MERGED INTO {}. {} REMAIN.'
#                              .format(self.reverse_vocab[c1] if c1 < len(self.reverse_vocab) else c1,
#                                      self.reverse_vocab[c2] if c2 < len(self.reverse_vocab) else c2,
#                                      self.cluster_counter,
#                                      len(self.current_batch) + len(word_queue) - 1))

    def corpus_generator(self):
        for tok in self.tokens:
            yield tok

    def create_index(self):
        corpus_iter1, corpus_iter2 = itertools.tee(self.corpus_generator())

        # increment one iterator to get consecutive tokens
        next(corpus_iter2)

        for w1, w2 in zip(corpus_iter1, corpus_iter2):
            if w1 in self.vocab and w2 in self.vocab:
                self.trans[self.vocab[w1]][self.vocab[w2]] += 1

        logging.info('{} word tokens were processed.'.format(self.num_tokens))

    def create_vocab(self):
        tmp_counts = defaultdict(int)
        for w in self.corpus_generator():
            tmp_counts[w] += 1
            self.num_tokens += 1

        words = sorted(tmp_counts.keys(), key=lambda w: tmp_counts[w],
                       reverse=True)

        too_rare = 0
        if self.max_vocab_size is not None \
           and len(words) > self.max_vocab_size:
            too_rare = tmp_counts[words[self.max_vocab_size]]
            if too_rare == tmp_counts[words[0]]:
                too_rare += 1
                logging.info("max_vocab_size too low.  Using all words that" +
                             " appeared > {} times.".format(too_rare))

        for i, w in enumerate(w for w in words if tmp_counts[w] > too_rare):
            self.vocab[w] = i
            self.counts[self.vocab[w]] = tmp_counts[w]

        self.reverse_vocab = sorted(self.vocab.keys(),
                                    key=lambda w: self.vocab[w])
        self.cluster_counter = len(self.vocab)

    def initialize_tables(self):
        logging.info("initializing tables")

        # edges between nodes
        total = 0
        for c1, c2 in itertools.combinations(self.current_batch, 2):
            w = self.compute_weight([c1], [c2]) \
                + self.compute_weight([c2], [c1])
            if w:
                self.w[c1][c2] = w
                total += w

        # edges to and from a single node
        for c in self.current_batch:
            w = self.compute_weight([c], [c])
            if w:
                self.w[c][c] = w
                total += w
        
        print(total)

        with tqdm(total=comb(len(self.current_batch), 2, exact=True), unit='pairs') as t:
            num_pairs = 0
            for c1, c2 in itertools.combinations(self.current_batch, 2):
                self.compute_L(c1, c2)
                num_pairs += 1
                t.update(1)
#                     logging.info("{} pairs precomputed".format(num_pairs))

    def compute_weight(self, nodes1, nodes2):
        paircount = 0
        for n1 in nodes1:
            for n2 in nodes2:
                paircount += self.trans[n1][n2]

        if not paircount:
            return 0.0

        count_1 = 0
        count_2 = 0
        for n in nodes1:
            count_1 += self.counts[n]
        for n in nodes2:
            count_2 += self.counts[n]

        # convert to floats
        num_tokens = float(self.num_tokens)
        paircount = float(paircount)
        count_1 = float(count_1)
        count_2 = float(count_2)

        return (paircount / num_tokens) \
               * log(paircount * num_tokens / count_1 / count_2, 2)

    def compute_L(self, c1, c2):
        val = 0.0

        # add the weight of edges coming in to the potential
        # new cluster from other nodes
        # TODO this is slow
        for d in self.current_batch:
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
        for d in self.current_batch:
            for c in [c1, c2]:
                if d in self.w[c]:
                    val -= self.w[c][d]
                elif c in self.w[d]:
                    val -= self.w[d][c]

        self.L[c1][c2] = val

    def find_best(self):
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
    
        return c1, c2

    def merge(self, c1, c2):
        c_new = self.cluster_counter

        # record parents
        self.cluster_parents[c1] = c_new
        self.cluster_parents[c2] = c_new
        r = random.randint(0, 1)
        self.cluster_bits[c1] = str(r)  # assign bits randomly
        self.cluster_bits[c2] = str(1 - r)

        # add the new cluster to the counts and transitions dictionaries
        self.counts[c_new] = self.counts[c1] + self.counts[c2]
        for c in [c1, c2]:
            for d, val in self.trans[c].items():
                if d == c1 or d == c2:
                    d = c_new
                self.trans[c_new][d] += val

        # subtract the weights for the merged nodes from the score table
        # TODO this is slow
        for c in [c1, c2]:
            for d1 in self.L:
                for d2 in self.L[d1]:
                    self.L[d1][d2] -= self.compute_weight([d1, d2], [c])
                    self.L[d1][d2] -= self.compute_weight([c], [d1, d2])

        # remove merged clusters from the counts and transitions dictionaries
        # to save memory (but keep frequencies for words for the final output)
        if c1 >= len(self.vocab):
            del self.counts[c1]
        if c2 >= len(self.vocab) and c2 in self.counts:
            del self.counts[c2]

        del self.trans[c1]
        del self.trans[c2]
        for d in self.trans:
            for c in [c1, c2]:
                if c in self.trans[d]:
                    del self.trans[d][c]

        # remove the old clusters from the w and L tables
        for table in [self.w, self.L]:
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
        self.current_batch.remove(c1)
        self.current_batch.remove(c2)

        # add the new cluster to the w and L tables
        self.add_to_batch(c_new)

    def add_to_batch(self, c_new):
        # compute weights for edges connected to the new node
        for d in self.current_batch:
            self.w[d][c_new] = self.compute_weight([d], [c_new])
            self.w[d][c_new] = self.compute_weight([c_new], [d])
        self.w[c_new][c_new] = self.compute_weight([c_new], [c_new])

        # add the weights from this new node to the merge score table
        # TODO this is slow
        for d1 in self.L:
            for d2 in self.L[d1]:
                self.L[d1][d2] += self.compute_weight([d1, d2], [c_new])
                self.L[d1][d2] += self.compute_weight([c_new], [d1, d2])

        # compute scores for merging it with all clusters in the current batch
        for d in self.current_batch:
            self.compute_L(d, c_new)

        # now add it to the batch
        self.current_batch.append(c_new)

    def get_bitstring(self, w):
        # walk up the cluster hierarchy until there is no parent cluster
        cur_cluster = self.vocab[w]
        bitstring = ""
        while cur_cluster in self.cluster_parents:
            bitstring = self.cluster_bits[cur_cluster] + bitstring
            cur_cluster = self.cluster_parents[cur_cluster]
        return bitstring

    def save_clusters(self, output_path):
        with open(output_path, 'w') as f:
            for w in self.vocab:
                # convert the counts back to ints when printing
                f.write("{}\t{}\t{}\n".format(w, self.get_bitstring(w),self.counts[self.vocab[w]]))
    
    def print_clusters(self):
        for w in self.vocab:
            print("{}\t{}\t{}".format(w, self.get_bitstring(w), self.counts[self.vocab[w]]))

english = './TEXTEN1.txt'
words = open_text(english)

c = ClassLMClusters(['<s>'] + words[:8000])