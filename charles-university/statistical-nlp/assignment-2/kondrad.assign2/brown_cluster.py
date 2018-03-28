from collections import defaultdict, Counter, Iterable
import itertools
import numpy as np
from tqdm import tqdm, trange
from scipy.special import comb


class LmCluster:
    """Implements the Brown clustering algorithm"""

    def __init__(self, words, word_cutoff=10):
        self.word_cutoff = word_cutoff

        # Process the text's word frequency distribution
        # Convert words to class numbers (integers)
        word_counts = Counter(words)
        word_set = sorted(word_counts, key=lambda w: word_counts[w], reverse=True)

        self.text_size = len(words)
        self.word2int = {}
        self.unigram_dist = defaultdict(int)
        for i, w in enumerate(word_set):
            self.word2int[w] = i
            self.unigram_dist[i] = word_counts[w]

        self.int2word = sorted(self.word2int, key=lambda word: self.word2int[word])
        self.unigrams = [self.word2int[w] for w in words]

        # Process the bigram distribution from unigrams
        self.bigrams = list(zip(self.unigrams, self.unigrams[1:]))
        self.bigram_dist = defaultdict(lambda: defaultdict(int))
        for wprev, w in self.bigrams:
            self.bigram_dist[wprev][w] += 1

        # Initialize each word in its own class, and only consider classes of words that appear frequently enough
        self.classes = [word for word in self.unigram_dist if self.unigram_dist[word] >= self.word_cutoff]
        self.class_counter = len(self.unigram_dist)

        # Keep track of merges in the clustering algorithm
        self.merge_history = []
        self.merge_tree = {}

        # Initialize sum and loss tables
        self.W = self.build_sum_table()
        self.L = self.build_loss_table()

    def build_sum_table(self):
        W = defaultdict(lambda: defaultdict(float))
        for l, r in itertools.combinations(self.bigram_dist, 2):
            W[l][r] = self.mutual_information([l], [r]) + self.mutual_information([r], [l])
        for c in self.bigram_dist:
            W[c][c] = self.mutual_information([c], [c])
        return W

    def build_loss_table(self):
        L = defaultdict(lambda: defaultdict(float))

        total = comb(len(self.classes), 2, exact=True)
        for l, r in tqdm(itertools.combinations(self.classes, 2), total=total, unit='pair'):
            L[l][r] = self.mi_loss(l, r)

        return L

    def mi_loss(self, l, r):
        mi = 0.0
        mi += self.mutual_information([l], [r])
        mi += self.mutual_information([r], [l])
        mi += self.mutual_information([l, r], [l, r])

        for c in self.bigram_dist:
            if c in [l, r]:
                continue
            mi += self.mutual_information([l, r], [c])
            mi += self.mutual_information([c], [l, r])

        for c in self.bigram_dist:
            for d in [l, r]:
                if c in self.W[d]:
                    mi -= self.W[d][c]
                if d in self.W[c]:
                    mi -= self.W[c][d]

        return mi

    def cluster(self, class_count=1):
        merges = len(self.classes) - class_count
        for _ in trange(merges, unit='class'):
            mi, l, r = self.best_merge()
            c_new = self.merge(l, r)

            save = (*self.class_name([l, r]), c_new, mi)
            self.merge_history.append(save)

    #             print(save)

    def best_merge(self):
        mi = ((self.L[l][r], l, r) for l in self.L for r in self.L[l])
        return max(mi, key=lambda x: x[0])

    def merge(self, l, r):
        c_new = self.class_counter

        self.unigram_dist[c_new] = self.unigram_dist[l] + self.unigram_dist[r]

        for c in [l, r]:
            for d, count in self.bigram_dist[c].items():
                d = c_new if d in [l, r] else d
                self.bigram_dist[c_new][d] += count

        for c in self.bigram_dist:
            for d in [l, r]:
                if d in self.bigram_dist[c] and c != c_new:
                    self.bigram_dist[c][c_new] += self.bigram_dist[c][d]

        for a in self.L:
            for b in self.L[a]:
                for c in [l, r]:
                    self.L[a][b] -= self.mutual_information([a, b], [c])
                    self.L[a][b] -= self.mutual_information([c], [a, b])

        del self.bigram_dist[l]
        del self.bigram_dist[r]
        for c in self.bigram_dist:
            for d in [l, r]:
                if d in self.bigram_dist[c]:
                    del self.bigram_dist[c][d]

        for table in [self.W, self.L]:
            for c in table:
                for d in [l, r]:
                    if d in table[c]:
                        del table[c][d]
            if l in table:
                del table[l]
            if r in table:
                del table[r]

        for c in [l, r]:
            self.classes.remove(c)

        for c in self.bigram_dist:
            if c == c_new:
                continue
            self.W[c][c_new] = self.mutual_information([c], [c_new]) + self.mutual_information([c_new], [c])
        self.W[c_new][c_new] = self.mutual_information([c_new], [c_new])

        for c in self.classes:
            self.L[c][c_new] = self.mi_loss(c, c_new)

        # Update classes
        self.classes.append(c_new)

        # Update the tree of classes
        self.merge_tree[l] = c_new
        self.merge_tree[r] = c_new

        self.class_counter += 1

        return c_new

    def mutual_information(self, left, right):
        bigram_count = np.sum(self.bigram_dist[l][r] for l in left for r in right)

        if not bigram_count:
            return 0.0

        left_count = np.sum(self.unigram_dist[c] for c in left)
        right_count = np.sum(self.unigram_dist[c] for c in right)

        mi = (bigram_count / self.text_size) * np.log2(bigram_count * self.text_size / left_count / right_count)
        return mi

    def class_name(self, classes):
        if not isinstance(classes, Iterable):
            classes = [classes]

        classes = [self.int2word[c] if c < len(self.int2word) else c for c in classes]
        return classes if len(classes) > 1 else classes[0]

    def get_classes(self):
        classes = defaultdict(list)

        for w in self.unigram_dist:
            if w not in self.merge_tree:
                if w in self.classes:
                    classes[w].append(w)
                continue

            c = w
            while c in self.merge_tree:
                c = self.merge_tree[c]

            classes[c].append(w)

        return classes


def open_text(filename):
    """Reads a text line by line, applies light preprocessing, and returns an array of words"""

    def preprocess(word):
        return word.strip()

    with open(filename, encoding='iso-8859-2') as f:
        content = f.readlines()

    return [preprocess(word) for word in content]


if __name__ == '__main__':
    english = './TEXTEN1.txt'
    words_en = open_text(english)

    lm = LmCluster(words_en[:8000])
    lm.cluster()