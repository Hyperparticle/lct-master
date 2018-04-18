#!/usr/bin/env python3
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin


def kmeans_cluster(X, n_clusters):
    """Compute the k-means algorithm on the given points"""

    # 1. Randomly choose center points for clusters
    perm = np.random.permutation(len(X))[:n_clusters]
    centers = X[perm]
    new_centers = None
    labels = None

    while not np.all(centers == new_centers):
        # 2.1 Assign labels based on closest center
        labels = kmeans_predict(X, centers)

        # 2.2 Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        centers = new_centers

    return centers, labels


def kmeans_predict(X, centers):
    return pairwise_distances_argmin(X, centers)


def plot_data(data, labels, centers):
    """Plot of K-means clustering taken from
    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html"""

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh.
    Z = kmeans_predict(np.c_[xx.ravel(), yy.ravel()], centers)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel1,
               aspect='auto', origin='lower')

    plt.scatter(data[:, 0], data[:, 1], s=5, c=labels, cmap=plt.cm.Dark2)
    # Plot the centroids as a white X
    centroids = centers
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='black', zorder=10)
    plt.title('K-means clustering')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def bench_k_means(name, t, gold, pred, data):
    """Compute k-means benchmark scores based on
    http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html"""

    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, t,
             metrics.homogeneity_score(gold, pred),
             metrics.completeness_score(gold, pred),
             metrics.v_measure_score(gold, pred),
             metrics.adjusted_rand_score(gold, pred),
             metrics.adjusted_mutual_info_score(gold, pred),
             metrics.silhouette_score(data, pred,
                                      metric='euclidean',
                                      sample_size=300)))


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

print("n_clusters: %d, \t n_samples %d, \t n_features %d" % (n_clusters, n_samples, n_features))
print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

# Do the clustering
t0 = time()
centers, __ = kmeans_cluster(x_train, n_clusters)
y_pred = kmeans_predict(x_test, centers)
t = time() - t0

bench_k_means("k-means", t, y_test, y_pred, x_test)
print(82 * '_')

plot_data(x_test, y_test, centers)
