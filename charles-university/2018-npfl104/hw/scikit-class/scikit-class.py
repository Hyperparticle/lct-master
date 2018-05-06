#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

import os
import gzip
import shutil


def vectorize(train, test):
    train_cols, test_cols = [], []

    combined = train.append(test)

    for column in train:
        lb = LabelBinarizer()
        lb.fit(combined[column])

        train_cols.append(lb.transform(train[column]))
        test_cols.append(lb.transform(test[column]))

    return np.concatenate(train_cols, axis=-1), np.concatenate(test_cols, axis=-1)


def evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    target_pred = model.predict(x_test)
    return accuracy_score(y_test, target_pred)


models = [
    [BernoulliNB(), 'naive bayes'],
    [Perceptron(max_iter=10), 'perceptron'],
    [DecisionTreeClassifier(), 'decision tree'],
    [AdaBoostClassifier(), 'ada boost'],
]

repo = '2018-npfl104-shared/data'

datasets = {}

for dataset_dir in sorted([d for d in os.listdir(repo) if os.path.isdir(os.path.join(repo, d))]):
    with gzip.open(os.path.join(repo, dataset_dir, 'train.txt.gz'), 'rb') as f_in:
        with open(os.path.join(repo, dataset_dir, 'train.txt'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    with gzip.open(os.path.join(repo, dataset_dir, 'test.txt.gz'), 'rb') as f_in:
        with open(os.path.join(repo, dataset_dir, 'test.txt'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    header = None if dataset_dir not in ['cinlp-twitter'] else 'infer'
    delimiter = ',' if dataset_dir not in ['pamap-easy'] else '\t'

    train = pd.read_csv(os.path.join(repo, dataset_dir, 'train.txt'), delimiter=delimiter, header=header)
    test = pd.read_csv(os.path.join(repo, dataset_dir, 'test.txt'), delimiter=delimiter, header=header)

    x_train, y_train = train[train.columns[:-1]], train[train.columns[-1:]]
    x_test, y_test = test[test.columns[:-1]], test[test.columns[-1:]]

    datasets[dataset_dir] = (x_train, y_train, x_test, y_test)


def process_dataset(name):
    if name == 'car-evaluation':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = x_train[x_train.columns[1:]], x_test[x_test.columns[1:]]

        x_train, x_test = vectorize(x_train, x_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'cinlp-twitter':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        x_train = x_train.drop(['Tweet', 'lemmas'], axis=1)
        x_test = x_test.drop(['Tweet', 'lemmas'], axis=1)

        return x_train, y_train, x_test, y_test
    elif name == 'connect-4-interpreted':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = vectorize(x_train, x_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'connect-4-raw':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = vectorize(x_train, x_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'credit-card-fraud':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'motion-capture-hand':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'mushrooms':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'music-genre-classification':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        x_train, x_test = x_train[x_train.columns[22:]], x_test[x_test.columns[22:]]

        return x_train, y_train, x_test, y_test
    elif name == 'pamap-easy':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'poker':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = vectorize(x_train, x_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'poker-with-extra-features':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = vectorize(x_train, x_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'spectf-heart':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'ctf-heart':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'wine-quality':
        x_train, y_train, x_test, y_test = datasets[name]
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test


for dataset in datasets:
    print()
    print(dataset)

    processed = process_dataset(dataset)

    if not processed:
        continue

    for model, name in models:
        result = evaluate(model, *processed)
        print(name, '{:.3f}'.format(result))
