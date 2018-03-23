from collections import defaultdict, Counter, Iterable
import itertools
import numpy as np
from tqdm import tqdm, trange
from scipy.special import comb


class LmCluster:
    def __init__(self, words, word_cutoff=10):
        self.word_cutoff = word_cutoff

        # Unigrams
        self.text_size = len(words)
        self.word2int = {}
        self.unigram_dist = defaultdict(int)

        word_counts = Counter(words)
        word_set = sorted(word_counts, key=lambda w: word_counts[w], reverse=True)

        for i, w in enumerate(word_set):
            self.word2int[w] = i
            self.unigram_dist[i] = word_counts[w]

        self.int2word = sorted(self.word2int, key=lambda word: self.word2int[word])
        self.unigrams = [self.word2int[w] for w in words]

        # Bigrams
        self.bigrams = list(zip(self.unigrams, self.unigrams[1:]))
        self.bigram_dist = defaultdict(lambda: defaultdict(int))
        for wprev, w in self.bigrams:
            self.bigram_dist[wprev][w] += 1

        self.classes = [word for word in self.unigram_dist if self.unigram_dist[word] >= self.word_cutoff]
        self.class_counter = len(self.unigram_dist)

        self.merge_history = []

    def cluster(self, class_count=1):
        merges = len(self.classes) - class_count
        for _ in trange(merges, unit='class'):
            mi, l, r = self.best_merge()
            c_new = self.merge(l, r)

            save = (*self.class_name([l, r]), c_new, mi)
            self.merge_history.append(save)

            print(save)

    def best_merge(self):
        s = defaultdict(float)
        for c in self.classes:
            s[c] += np.sum(self.mutual_information([a], [c]) for a in self.bigram_dist)
            s[c] += np.sum(self.mutual_information([c], [b]) for b in self.bigram_dist)
            s[c] -= self.mutual_information([c], [c])

        mi = (self.merge_mi(l, r, s) for l, r in itertools.combinations(self.classes, 2))
        progress = tqdm(mi, total=comb(len(self.classes), 2, exact=True), leave=False)
        return max(progress, key=lambda x: x[0])

    def merge_mi(self, l, r, s):
        mi = 0.0
        mi -= s[l] + s[r]
        mi += self.mutual_information([l], [r])
        mi += self.mutual_information([r], [l])
        mi += self.mutual_information([l, r], [l, r])

        for c in self.bigram_dist:
            if c in [l, r]:
                continue
            mi += self.mutual_information([l, r], [c])
            mi += self.mutual_information([c], [l, r])

        return mi, l, r

    def merge(self, l, r):
        c_new = self.class_counter

        # Add the new class to frequency distributions
        self.unigram_dist[c_new] = self.unigram_dist[l] + self.unigram_dist[r]

        for c in [l, r]:
            for d, count in self.bigram_dist[c].items():
                d = c_new if d in [l, r] else d
                self.bigram_dist[c_new][d] += count

        for c in self.bigram_dist:
            for d in [l, r]:
                if d in self.bigram_dist[c] and c != c_new:
                    self.bigram_dist[c][c_new] += self.bigram_dist[c][d]

        #         del self.bigram_dist[l]
        #         del self.bigram_dist[r]
        #         for c in self.bigram_dist:
        #             for d in [l, r]:
        #                 if d in self.bigram_dist[c]:
        #                     del self.bigram_dist[c][d]

        # Update classes
        for c in [l, r]:
            self.classes.remove(c)
        self.classes.append(c_new)

        self.class_counter += 1

        return c_new

    def mutual_information_total(self):
        return np.sum(
            self.mutual_information([wprev], [w]) for wprev in self.bigram_dist for w in self.bigram_dist[wprev])

    def mutual_information(self, left, right):
        bigram_count = np.sum(self.bigram_dist[l][r] for l in left for r in right)

        if not bigram_count:
            return 0.0

        left_count = np.sum(self.unigram_dist[c] for c in left)
        right_count = np.sum(self.unigram_dist[c] for c in right)

        return (bigram_count / self.text_size) * np.log2(bigram_count * self.text_size / left_count / right_count)

    def class_name(self, classes):
        if not isinstance(classes, Iterable):
            classes = [classes]

        classes = [self.int2word[c] if c < len(self.int2word) else c for c in classes]
        return classes if len(classes) > 1 else classes[0]


def preprocess(word):
    return word.strip()


def open_text(filename):
    """Reads a text line by line, applies light preprocessing, and returns an array of words"""
    with open(filename, encoding='iso-8859-2') as f:
        content = f.readlines()

    return [preprocess(word) for word in content]


if __name__ == '__main__':
    english = './TEXTEN1.txt'
    words_en = open_text(english)

    lm = LanguageModel(words_en[:8000])
    lm.cluster(15)
