#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


class LinearRegression:
    def __init__(self, alpha=1e-5, iterations=5000):
        self.alpha = alpha
        self.iterations = iterations

        self.coef, self.intercept = None, None

    def fit(self, X, y):
        assert len(X) == len(y)

        y = np.expand_dims(y, axis=1)
        n, m = X.shape
        self.W = np.random.normal(size=[m, 1])

        for i in range(self.iterations):
            pred = self._func(X)
            error = pred - y
            W_grad = X.T.dot(error) / n
            self.W -= self.alpha * W_grad

    def predict(self, X):
        return self._func(X)

    def _func(self, X):
        return X.dot(self.W)


def evaluate(model, train, test):
    x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

    model.fit(x_train, y_train)
    target_pred = model.predict(x_test)

    mse = ((target_pred - y_test) ** 2).mean()
    return mse


def vectorize(train, test):
    v = DictVectorizer(sparse=False)
    d = train.append(test).to_dict('records')
    x = v.fit_transform(d)
    return x[:len(train)], x[len(train):]


np.random.seed(100)

model = LinearRegression()

train = np.loadtxt('artificial_2x_train.tsv', delimiter='\t')
test = np.loadtxt('artificial_2x_test.tsv', delimiter='\t')

result = evaluate(model, train, test)
print('artificial_2x')
print('MSE: {:.3f}'.format(result))
print()


cols = [i for i in range(9)]
train = pd.read_csv('pragueestateprices_train.tsv', delimiter='\t', usecols=cols, header=None)
test = pd.read_csv('pragueestateprices_test.tsv', delimiter='\t', usecols=cols, header=None)
train.columns = test.columns = [str(i) for i in range(9)]

train, test = vectorize(train, test)

result = evaluate(model, train, test)
print('pragueestateprices')
print('MSE: {:.3f}'.format(result))
print()
