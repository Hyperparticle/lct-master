#!/usr/bin/env python3

import pandas as pd
import numpy as np
from helper import train_test_split

def predict(row, weights):
    weighted_sum = weights[0] + np.dot(weights[1:], row[:-1])
    return 1 if weighted_sum >= 0 else 0

def train_weights(train, learn_rate, epochs):
    weights = np.zeros_like(train[0])

    for epoch in range(epochs):
        for row in train:
            error = row[-1] - predict(row, weights)
            weights[0] += learn_rate * error
            for i in range(len(row)-1):
                weights[i + 1] += learn_rate * error * row[i]

    return weights

def accuracy(data, weights):
    predicted = [predict(row, weights) for row in data]
    actual = [row[-1] for row in data]
    return sum(1 for p,a in zip(predicted, actual) if p == a) / len(data) * 100.0

train, test = train_test_split()

weights = train_weights(train['artificial'], 0.1, 5)
acc = accuracy(test['artificial'], weights)
print(acc)

weights = train_weights(train['income'], 0.1, 5)
acc = accuracy(test['income'], weights)
print(acc)
