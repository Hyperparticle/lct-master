#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
from collections import Counter
from helper import train_test_split

def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value: left.append(row)
        else: right.append(row)
    return left, right
 
def gini_index(groups, classes):
    n = sum(len(group) for group in groups)
    gini = 0
    for group in groups:
        if len(group) == 0:
            continue
        score = 0
        for c in classes:
            p = [row[-1] for row in group].count(c) / len(group)
            score += p ** 2
        gini += (1 - score) * len(group) / n
    return gini

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 1e10, 1e10, 1e10, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

def terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth):
    left, right = node['groups']
    del node['groups']
    if not left or not right:
        node['left'] = node['right'] = terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = terminal(left)
    else:
        node['left'] = get_split(left)
        split_node(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = terminal(right)
    else:
        node['right'] = get_split(right)
        split_node(node['right'], max_depth, min_size, depth+1)

def predict(node, row):
    if row[node['index']] < node['value']:
        return predict(node['left'], row) if isinstance(node['left'], dict) else node['left']
    else:
        return predict(node['right'], row) if isinstance(node['right'], dict) else node['right']

def accuracy(train, test):
    tree = get_split(train)
    split_node(tree, 3, 5, 1)
    predicted = [predict(tree, row) for row in test]
    actual = [row[-1] for row in test]
    return sum(1 for p,a in zip(predicted, actual) if p == a) / len(test) * 100.0

train, test = train_test_split()

print(accuracy(train['artificial'], test['artificial']))
print(accuracy(train['income'][:500], test['income']))
