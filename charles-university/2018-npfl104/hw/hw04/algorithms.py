#!/usr/bin/env python3
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from helper import train_test_split

train, test = train_test_split()

def evaluate(model, dataset):
    model.fit(train[dataset][:,:-1], train[dataset][:,-1])
    target_pred = model.predict(test[dataset][:,:-1])
    return accuracy_score(test[dataset][:,-1], target_pred, normalize = True)

nb = BernoulliNB()
lr = LogisticRegression()
perc = Perceptron(max_iter=5)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svc = SVC(max_iter=5)

models = [nb, lr, perc, dt, knn, svc]

for model in models:
    print(evaluate(model, 'artificial'))
    print(evaluate(model, 'income'))
