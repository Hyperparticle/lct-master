#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


np.random.seed(60)

# Load the data
train = np.loadtxt('pamap_easy.train.txt', delimiter='\t')
test = np.loadtxt('pamap_easy.test.txt', delimiter='\t')

x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

# Normalize the data
x_train = x_train / x_train.max(axis=0) * 100
x_test = x_test / x_test.max(axis=0) * 100

n_samples, n_features = train.shape
n_clusters = len(np.unique(y_train))

param_grid = {
    'C': [1, 5, 10, 50, 100],
    'gamma': [0.1, 0.5, 1, 5, 10],
}

for kernel in ['rbf', 'linear', 'poly']:
    print(kernel)

    clf = GridSearchCV(SVC(kernel=kernel), param_grid, scoring='accuracy', n_jobs=8)
    clf.fit(x_train, y_train)
    print(clf.best_params_)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    if kernel == 'rbf':
        scores = [x[1] for x in clf.grid_scores_]
        scores = np.array(scores).reshape(len(param_grid['C']), len(param_grid['gamma']))

        df = pd.DataFrame(scores, columns=param_grid['gamma'])
        print(df)
        matrix = df.as_matrix()

        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='hot')

        ax.set_xticks(np.arange(len(param_grid['gamma'])))
        ax.set_yticks(np.arange(len(param_grid['C'])))
        ax.set_xticklabels(param_grid['gamma'])
        ax.set_yticklabels(param_grid['C'])
        for i in range(len(param_grid['C'])):
            for j in range(len(param_grid['gamma'])):
                text = ax.text(j, i, '{:.3f}'.format(matrix[i, j]), ha="center", va="center", color="g",
                               fontsize='smaller')
        ax.set_title("Heatmap of C and gamma parameters in RBF SVC")
        ax.set_xlabel('gamma')
        ax.set_ylabel('C')
        fig.tight_layout()
        plt.savefig('heatmap.png')
    print()
