#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer

def evaluate(model, train, test):
    model.fit(train[:,:-1], train[:,-1])
    target_pred = model.predict(test[:,:-1])
    actual = test[:,-1]
    return np.sum(np.square(target_pred - actual))

models = [
    [LinearRegression(), 'linear regression'],
]

# train = np.loadtxt('artificial_2x_train.tsv', delimiter='\t')
# test = np.loadtxt('artificial_2x_test.tsv', delimiter='\t')
# train = np.loadtxt('pragueestateprices_train.tsv', delimiter='\t')
# test = np.loadtxt('pragueestateprices_test.tsv', delimiter='\t')


# for model, name in models:
#     print(name, evaluate(model, train, test))

v = DictVectorizer(sparse=False)
train = pd.read_csv('pragueestateprices_train.tsv', delimiter='\t')
print()



D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
print(X)