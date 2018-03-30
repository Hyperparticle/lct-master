import numpy as np
import pandas as pd
import collections as c
import nltk
from sklearn.metrics import accuracy_score
import itertools
import dill as pickle

from subprocess import call


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


def evaluate_hmm(split, i=0, lang='', unsupervised=False, load=False):
    train, heldout, test = split

    name = 'unsupervised' if unsupervised else 'supervised'
    filename = 'data/hmm_{}_tagger_{}_{}.pkl'.format(name, lang, i)

    if unsupervised:
        labeled = sentence_split(train[:10_000])
        unlabeled = sentence_split(train[10_000:])
    else:
        labeled = sentence_split(train)

    words, tags = list(zip(*(train + heldout + test)))
    states, symbols = list(set(tags)), list(set(words))

    test = sentence_split(test)

    trainer = nltk.hmm.HiddenMarkovModelTrainer(states, symbols)

    print('Evaluating HMM {} {} [{}]'.format(name, lang, i))
    if load:
        with open(filename, 'rb') as f:
            tagger = pickle.load(f)
    else:
        tagger = trainer.train_supervised(labeled,
                                          estimator=lambda fd, bins: nltk.probability.LidstoneProbDist(fd, 0.1, bins))
        if unsupervised:
            tagger = trainer.train_unsupervised(unlabeled, model=tagger, max_iterations=5)
        with open(filename, 'wb') as f:
            pickle.dump(tagger, f)

    return tagger.evaluate(test)


# Read the texts into memory
english = './data/texten2.ptg'
czech = './data/textcz2.ptg'

words_en = open_text(english)
words_cz = open_text(czech)

splits_en = split_all(words_en)
splits_cz = split_all(words_cz)


train, heldout, test = splits_en[0]

words, tags = list(zip(*(train + heldout + test)))
states, symbols = list(set(tags)), list(set(words))

test = sentence_split(test)
trainer = nltk.hmm.HiddenMarkovModelTrainer(states, symbols)

labeled = sentence_split(train[:10_000])
unlabeled = sentence_split(train[10_000:])

tagger = trainer.train_supervised(labeled, estimator=lambda fd, bins: nltk.probability.LidstoneProbDist(fd, 0.1, bins))

print('\n')

print(tagger.evaluate(test))

for _ in range(5):
    tagger = trainer.train_unsupervised(unlabeled, model=tagger, max_iterations=1)
    print(tagger.evaluate(test))
