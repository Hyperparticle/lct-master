#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
from collections import Counter
from helper import train_test_split

np.random.seed = 100

def dist(x, y, length):
    # Euclidean distance
    return np.sqrt(np.add.reduce((x[:length] - y[:length]) ** 2))

def neighbors(train, test, k):
    sample = train[np.random.choice(len(train), 300)]
    distances = np.array(sorted([(x, dist(test, x, len(test))) for x in sample], key=lambda x: x[1]))[:k, 0]
    return distances

def prediction(nb):
    return Counter([n[-1] for n in nb]).most_common()[0][0]

def accuracy(train, test, k=2):
    nbs = np.array([neighbors(train, row, k) for row in test])
    predicted = [prediction(nb) for nb in nbs]
    actual = [row[-1] for row in test]
    return sum(1 for p,a in zip(predicted, actual) if p == a) / len(test) * 100.0

train, test = train_test_split()

print(accuracy(train['artificial'], test['artificial'], 6))
print(accuracy(train['income'], test['income'], 100))
