#!/usr/bin/env python3

import numpy as np
from collections import Counter
from collections import defaultdict
import sys
import pickle
import re

if (len(sys.argv) < 2):
    print("Usage: generate.py [filename] [n]")
    exit()

n = int(sys.argv[2]) if len(sys.argv) >= 3 else 20

file = open(sys.argv[1])
text = file.read()

# Process tokens
sentences = re.split(r'[.:;]\s+', text.lower()) # lowercase, split on sentence boundaries
sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences] # Remove additional punctuation
sentences = ['<s> ' + sentence + ' </s>' for sentence in sentences] # Add sentence boundary tokens
tokens = [w for sentence in sentences for w in sentence.split()] # Split on whitespace

counts = Counter(tokens)

bigrams = defaultdict(Counter)
for wprev,w in zip(tokens, tokens[1:]):
    bigrams[wprev][w] += 1

# Save frequency distribution
with open('counts.pkl', 'wb') as f:
    pickle.dump(counts, f)
with open('bigrams.pkl', 'wb') as f:
    pickle.dump(bigrams, f)

with open('counts.pkl', 'rb') as f:
    counts = pickle.load(f)
with open('bigrams.pkl', 'rb') as f:
    bigrams = pickle.load(f)

def most_freq_seq(n):
    words = ['<s>'] # Start with the beginning of a sentence

    for _ in range(n):
        word = words[-1]
        next_words = list(bigrams[word].keys())

        # No next word -> exit loop
        if (len(next_words) == 0):
            break

        probs = [bigrams[word][n] / sum(bigrams[word].values()) for n in next_words]
        words.append(np.random.choice(next_words, p=probs))

        # Reached the end of a sentence
        if (words[-1] == '</s>'):
            words = words[:-1]
            break

    return words[1:]

# Print results
print(' '.join(w for w in most_freq_seq(n)))
