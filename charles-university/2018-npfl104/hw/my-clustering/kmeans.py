#!/usr/bin/env python3
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans

np.random.seed(42)

train = np.loadtxt('pamap_easy.train.txt', delimiter='\t')
test = np.loadtxt('pamap_easy.test.txt', delimiter='\t')

x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

n_samples, n_features = train.shape
n_clusters = len(np.unique(y_train))

print("n_clusters: %d, \t n_samples %d, \t n_features %d" % (n_clusters, n_samples, n_features))
print(82 * '_')
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(name, t, gold, pred, data):
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


t0 = time()
kmeans = KMeans(n_clusters)
kmeans.fit(x_train)
y_pred = kmeans.predict(x_test)
t = time() - t0

bench_k_means("k-means++", t, y_test, y_pred, x_test)
print(82 * '_')


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
reduced_data = x_test
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c=y_test)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
