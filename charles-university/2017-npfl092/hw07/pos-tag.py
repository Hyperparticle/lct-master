#!/usr/bin/env python3

import sys
import urllib.request
from nltk.tag import tnt

text_devel = 'tagger-devel.tsv'
text_eval = 'tagger-eval.tsv'

def download(url, filename):
    response = urllib.request.urlopen(url)
    content = response.read().decode('iso-8859-2')
    with open(filename, 'w') as f:
        f.write(content)

# Download train/test files
download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-devel.tsv', text_devel)
download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-eval.tsv', text_eval)

# Train
tagger = tnt.TnT()
with open(text_devel, 'r') as f:
    tags = [line.split() for line in f]
    tags = [[tag for tag in tags if len(tag) == 2]]
    tagger.train(tags)

# Test
with open(text_eval, 'r') as f:
    correct, total = 0, 0

    tags = [line.split() for line in f]
    tags = [tag for tag in tags if len(tag) == 2]
    for word,answer in tags:
        predicted = tagger.tag([word])[0][1]
        total += 1
        if answer == predicted:
            correct += 1
    
    print("accuracy: " + str(correct) + "/" + str(total) + " = " + str(correct / total))