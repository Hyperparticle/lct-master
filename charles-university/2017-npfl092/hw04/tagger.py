#!/usr/bin/env python3

from collections import Counter
from collections import defaultdict
import pickle

class Tagger:
    def __init__(self):
        self.model = defaultdict(Counter)

    def see(self, word, pos):
        word = word.lower()
        self.model[word].update(pos)
    
    def train(self):
        pass
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, word):
        word = word.lower()
        counts = self.model
        counts[word].most_common(1)[0][0] if word in counts else ""
