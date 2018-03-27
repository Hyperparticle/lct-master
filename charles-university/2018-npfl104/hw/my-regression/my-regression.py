#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def evaluate(model, train, test):
    model.fit(train[:,:-1], train[:,-1])
    target_pred = model.predict(test[:,:-1])
    actual = test[:,-1]
    return np.sum(np.square(target_pred - actual))

models = [
    [LinearRegression(), 'linear regression'],
]

train = np.loadtxt('artificial_2x_train.tsv', delimiter='\t')
test = np.loadtxt('artificial_2x_test.tsv', delimiter='\t')

for model, name in models:
    print(name, evaluate(model, train, test))