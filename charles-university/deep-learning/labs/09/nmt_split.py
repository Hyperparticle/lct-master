#!/usr/bin/env python3

import morpho_dataset

train = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-train.txt")
dev = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-dev.txt")
test = morpho_dataset.MorphoDataset("czech-pdt/czech-pdt-test.txt")

vocab = set()

for dataset, name in [[train, 'train'], [dev, 'dev'], [test, 'test']]:
    with open('czech-pdt-nmt/czech-pdt-nmt-' + name + '-source.txt', 'w') as source, \
         open('czech-pdt-nmt/czech-pdt-nmt-' + name + '-target.txt', 'w') as target:
        w = dataset.factors[dataset.FORMS].words
        l = dataset.factors[dataset.LEMMAS].words
        for word_sent, lemma_sent in zip(dataset.factors[dataset.FORMS].word_ids, dataset.factors[dataset.LEMMAS].word_ids):
            for word, lemma in zip(word_sent, lemma_sent):
                print(' '.join(w[word]), file=source)
                print(' '.join(l[lemma]), file=target)

                vocab.update(c for c in w[word])
                vocab.update(c for c in l[lemma])

for v in sorted(vocab):
    print(v)
