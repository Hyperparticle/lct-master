#!/usr/bin/env python3

import morpho_dataset

train = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-train.txt")
dev = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-dev.txt")
test = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-test.txt")

vocab = set()

for dataset in [train]:
    w = dataset.factors[dataset.FORMS].words
    l = dataset.factors[dataset.LEMMAS].words
    for word_sent, lemma_sent in zip(dataset.factors[dataset.FORMS].word_ids, dataset.factors[dataset.LEMMAS].word_ids):
        for word, lemma in zip(word_sent, lemma_sent):
            print('\t'.join([w[word], l[lemma], '_']))
        print()