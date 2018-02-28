#!/usr/bin/env python3
import numpy as np
from collections import Counter

if __name__ == "__main__":
    data_count = Counter()

    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line
            data_count.update(line)

    # TODO: Create a NumPy array containing the data distribution
    chars = sorted(data_count.keys())

    data_probs = np.array([data_count[k] / sum(data_count.values()) for k in chars])

    model_probs = {}

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line
            word, prob = line.split('\t')
            model_probs[word] = float(prob)

    model_probs = np.array([model_probs[k] for k in chars])

    entropy = - np.sum([p * np.log(p) for p in data_probs])

    # TODO: Compute and print entropy H(data distribution)
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = - np.sum([p * np.log(q) for (p,q) in zip(data_probs, model_probs)])
    print("{:.2f}".format(cross_entropy))

    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(kl_divergence))
