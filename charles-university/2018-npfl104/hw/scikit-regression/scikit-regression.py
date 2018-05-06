#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def vectorize(train, test):
    v = DictVectorizer(sparse=False)
    d = train.append(test).to_dict('records')
    x = v.fit_transform(d)
    return x[:len(train)], x[len(train):]


def evaluate(models, train, test):
    x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

    for model, name in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        print(name)
        print('MSE: {:.3f}'.format(mse))
        print()


np.random.seed(100)

models = [
    [LinearRegression(), 'linear regressor'],
    [KNeighborsRegressor(), 'k-nearest neighbors regressor'],
    [SVR(), 'support vector regressor'],
    [DecisionTreeRegressor(), 'decision tree regressor']
]

train = np.loadtxt('artificial_2x_train.tsv', delimiter='\t')
test = np.loadtxt('artificial_2x_test.tsv', delimiter='\t')

print('artificial_2x')
evaluate(models, train, test)


cols = [i for i in range(9)]
train = pd.read_csv('pragueestateprices_train.tsv', delimiter='\t', usecols=cols, header=None)
test = pd.read_csv('pragueestateprices_test.tsv', delimiter='\t', usecols=cols, header=None)
train.columns = test.columns = [str(i) for i in range(9)]

train, test = vectorize(train, test)

print('pragueestateprices')
evaluate(models, train, test)
