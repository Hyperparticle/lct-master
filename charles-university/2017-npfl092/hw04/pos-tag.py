#!/usr/bin/env python3

import sys
import urllib.request
from tagger import Tagger

text_devel = 'tagger-devel.tsv'
text_eval = 'tagger-eval.tsv'
model = 'model.pkl'

# argparse

def download(url, filename):
    response = urllib.request.urlopen(url)
    content = response.read().decode('iso-8859-2')
    with open(filename, 'w') as f:
        f.write(content)

if (len(sys.argv) != 2):
    print('Usage: ')
    exit()

cmd = sys.argv[1]

if (cmd == 'download'):
    download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-devel.tsv', text_devel)
    download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-eval.tsv', text_eval)
elif (cmd == 'train'):
    tagger = Tagger()

    with open(text_devel, 'r') as f:
        for line in f:
            s = line.split()
            if (len(s) != 2):
                continue
            word, tag = s
            tagger.see(word, tag)
        tagger.train()
        tagger.save(model)
elif (cmd == 'predict'):
    tagger = Tagger()
    tagger.load(model)
    lines = []

    with open(text_eval, 'r') as f:
        for line in f:
            s = line.split()

            if (len(s) < 2):
                lines.append("\n")
                continue

            word, tag = s
            predicted = tagger.predict(word)

            lines.append(word + "\t" + tag + "\t" + predicted + "\n")
    with open(text_eval, 'w') as f:
        for line in lines:
            f.write(line)
elif (cmd == 'eval'):
    correct, total = 0, 0

    with open(text_eval, 'r') as f:
        for line in f:
            s = line.split()

            if (len(s) != 3):
                continue

            word, tag, predicted = s

            total += 1
            if tag == predicted:
                correct += 1
    
    print("accuracy: " + str(correct) + "/" + str(total) + " = " + str(correct / total))