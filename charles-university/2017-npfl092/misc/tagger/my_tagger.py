#!/usr/bin/env python3

# iconv -t utf8 -f iso-8859-2 < 1.txt > 2.txt

import sys
from collections import Counter
from collections import defaultdict
import pickle

train = len(sys.argv) == 2 and sys.argv[1] == "-t"

counts = defaultdict(Counter)

if train:
    for line in sys.stdin:
        s = line.split()

        if (len(s) != 2):
            continue

        word, tag = s
        counts[word.lower()].update(tag)

    with open('train.pkl', 'wb') as f:
        pickle.dump(counts, f)
else:
    with open('train.pkl', 'rb') as f:
        counts = pickle.load(f)
    
    for line in sys.stdin:
        s = line.split()

        if (len(s) != 2):
            print()
            continue

        word, tag = s

        best_tag = counts[word.lower()].most_common(1)[0][0] if word.lower() in counts else "N"

        print(word + "\t" + tag + "\t" + best_tag)
