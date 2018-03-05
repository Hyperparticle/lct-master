#!/usr/bin/env python3
import numpy as np
from collections import Counter, defaultdict

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        data_count = Counter(line.rstrip('\n') for line in data.readlines())

    # Create a NumPy array containing the data distribution
    # chars = sorted(data_count.keys())
    # data_probs = np.array([data_count[k] / sum(data_count.values()) for k in chars])
    data_probs = defaultdict(int)
    data_probs.update({k:(data_count[k] / sum(data_count.values())) for k in data_count.keys()})

    model_probs = defaultdict(int)

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # Process the line
            word, prob = line.split('\t')
            model_probs[word] = float(prob)

    # model_probs = np.array([model_probs[k] for k in chars])

    entropy = - np.sum([data_probs[k] * np.log(data_probs[k]) for k in data_probs])

    # Compute and print entropy H(data distribution)
    print("{:.2f}".format(entropy))

    # Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = - np.sum([data_probs[k] * np.log(model_probs[k]) for k in data_probs])
    print("{:.2f}".format(cross_entropy))

    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(kl_divergence))
