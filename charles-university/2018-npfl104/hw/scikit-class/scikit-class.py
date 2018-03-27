#!/usr/bin/env python3
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def evaluate(model, train, test):
    model.fit(train[:,:-1], train[:,-1])
    target_pred = model.predict(test[:,:-1])
    return accuracy_score(test[:,-1], target_pred, normalize=True)

models = [
    [BernoulliNB(), 'naive bayes'],
    [Perceptron(max_iter=10), 'perceptron'],
    [DecisionTreeClassifier(), 'decision tree'],
    [KNeighborsClassifier(), 'k nearest neighbors'],
]

train = np.loadtxt('train.txt', delimiter=',')
test = np.loadtxt('test.txt', delimiter=',')

for model, name in models:
    print(name, evaluate(model, train, test))