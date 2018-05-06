#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

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


# def label(train, test):
#     combined = train.append(test)
#
#     le = LabelEncoder()
#     le.fit(combined.as_matrix().flatten())
#
#     return le.transform(train), le.transform(test)


def evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    target_pred = model.predict(x_test)
    return accuracy_score(y_test, target_pred)


models = [
    [BernoulliNB(), 'naive bayes'],
    [Perceptron(max_iter=10), 'perceptron'],
    [DecisionTreeClassifier(), 'decision tree'],
    [KNeighborsClassifier(), 'k nearest neighbors'],
]

repo = '2018-npfl104-shared/data'

# datasets = [
#     'car-evaluation',
#     'cinlp-twitter',
#     'connect-4-interpreted',
#     'connect-4-raw',
#     'credit-card-fraud',
#     'motion-capture-hand',
#     'mushrooms',
#     'music-genre-classification',
#     'music-genre-classification/img',
#     'map-easy',
#     'oker',
#     'oker-with-extra-features',
#     'ctf-heart',
#     'wine-quality',
# ]

datasets = {}

for dataset_dir in sorted([d for d in os.listdir(repo) if os.path.isdir(os.path.join(repo, d))]):
    # with gzip.open(os.path.join(repo, dataset_dir, 'train.txt.gz'), 'rb') as f_in:
    #     with open(os.path.join(repo, dataset_dir, 'train.txt'), 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    # with gzip.open(os.path.join(repo, dataset_dir, 'test.txt.gz'), 'rb') as f_in:
    #     with open(os.path.join(repo, dataset_dir, 'test.txt'), 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)

    header = None if dataset_dir not in ['cinlp-twitter'] else 'infer'

    train = pd.read_csv(os.path.join(repo, dataset_dir, 'train.txt'), delimiter=',', header=header)
    test = pd.read_csv(os.path.join(repo, dataset_dir, 'test.txt'), delimiter=',', header=header)

    x_train, y_train = train[train.columns[:-1]], train[train.columns[-1:]]
    x_test, y_test = test[test.columns[:-1]], test[test.columns[-1:]]

    datasets[dataset_dir] = (x_train, y_train, x_test, y_test)


def process_dataset(name):
    if name == 'car-evaluation':
        x_train, y_train, x_test, y_test = datasets[name]
        x_train, x_test = x_train[x_train.columns[1:]], x_test[x_test.columns[1:]]

        x_train, x_test = vectorize(x_train, x_test)
        # y_train, y_test = label(y_train, y_test)
        y_train, y_test = y_train.as_matrix().flatten(), y_test.as_matrix().flatten()

        return x_train, y_train, x_test, y_test
    elif name == 'cinlp-twitter':
    elif name == 'connect-4-interpreted':
    elif name == 'connect-4-raw':
    elif name == 'credit-card-fraud':
    elif name == 'motion-capture-hand':
    elif name == 'mushrooms':
    elif name == 'music-genre-classification':
    elif name == 'music-genre-classification/img':
    elif name == 'map-easy':
    elif name == 'oker':
    elif name == 'oker-with-extra-features':
    elif name == 'ctf-heart':
    elif name == 'wine-quality':


for dataset in datasets:
    print(dataset)

    processed = process_dataset(dataset)

    if not processed:
        continue

    for model, name in models:
        result = evaluate(model, *processed)
        print(name, '{:.3f}'.format(result))
    print()
