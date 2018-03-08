#!/usr/bin/env python3

import pandas as pd
import numpy as np

def to_int(col):
    values = np.unique(col.astype(str), return_inverse=True)[1]
    if len(set(values)) > 2:
        values = list(np.eye(np.max(values) + 1)[values])
    else:
        values = [[v] for v in values]
    return values

def flatten(df):
    return np.array([[x for item in row for x in item] for row in df.as_matrix()])

def train_test_split():
    data_train, data_test = {}, {}

    train = pd.read_csv('data/artificial_separable_train.csv', names=['size', 'color', 'shape', 'class'])
    train = train.apply(to_int)
    test = pd.read_csv('data/artificial_separable_test.csv', names=['size', 'color', 'shape', 'class'])
    test = test.apply(to_int)
    data_train['artificial'] = flatten(train)
    data_test['artificial'] = flatten(test)

    train = pd.read_csv('data/adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'])
    train[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']] = train[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']].apply(to_int)
    train[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = train[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].apply(lambda col: [[c] for c in col])
    test = pd.read_csv('data/adult.test', header=0, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'])
    test[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = test[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].apply(lambda col: [[c] for c in col])
    test[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']] = test[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']].apply(to_int)
    data_train['income'] = flatten(train)
    data_test['income'] = flatten(test)

    return data_train, data_test

