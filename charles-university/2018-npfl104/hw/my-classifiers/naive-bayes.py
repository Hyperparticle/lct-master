import pandas as pd
import numpy as np
import math
from helper import train_test_split

train, test = train_test_split()

def class_dict(data):
    classes = {}
    for row in data:
        if (row[-1] not in classes):
            classes[row[-1]] = []
        classes[row[-1]].append(row)
    return classes

def mean_std(data):
    return [(np.mean(col), np.std(col)) for col in list(zip(*data))[:-1]]

def mean_std_classes(data):
    classes = class_dict(data)
    mstd = {}
    for c in classes:
        mstd[c] = mean_std(classes[c])
    return mstd

def prob(x, mean, std):
    if std == 0.0: return 0
    return (1/(math.sqrt(2*math.pi)*std))*math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))

def prob_classes(mstd, row):
    p = {}
    for c in mstd:
        p[c] = np.multiply.reduce([
            prob(x, mean, std)
            for (mean, std), x in zip(mstd[c], row)])
    return p

def predict(mstd, row):
    probs = prob_classes(mstd, row)
    best = None, -1
    for c in probs:
        if best[0] is None or probs[c] > best[1]:
            best = c, probs[c]
    return best[0]

def accuracy(train, test):
    dist = mean_std_classes(train)
    predicted = [predict(dist, row) for row in test]
    actual = [row[-1] for row in test]
    return sum(1 for p,a in zip(predicted, actual) if p == a) / len(test) * 100.0

print(accuracy(train['artificial'], test['artificial']))
print(accuracy(train['income'], test['income']))
