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
    test = pd.read_csv('data/artificial_separable_test.csv', names=['size', 'color', 'shape', 'class'])
    
    traintest = pd.concat([train, test]).apply(to_int)
    
    data_train['artificial'] = flatten(traintest.iloc[:len(train)])
    data_test['artificial'] = flatten(traintest.iloc[len(train):])

    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'class']
    real = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    train = pd.read_csv('data/adult.data', names=columns)
    test = pd.read_csv('data/adult.test', header=0, names=columns)

    traintest = pd.concat([train, test])
    traintest['class'] = [c.rstrip('.') for c in traintest['class']]
    traintest[categorical] = traintest[categorical].apply(to_int)
    traintest[real] = traintest[real].apply(lambda col: [[c] for c in col])
    
    data_train['income'] = flatten(traintest.iloc[:len(train)])
    data_test['income'] = flatten(traintest.iloc[len(train):])

    return data_train, data_test
