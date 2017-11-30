#!/usr/bin/env python3

import sys
import urllib.request
import tagger

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
    download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-devel.tsv', 'tagger-devel.tsv')

    download('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/premium_tagger/tagger-eval.tsv', 'tagger-eval.tsv')
# elif (cmd == 'train'):

# elif (cmd == 'predict'):

# elif (cmd == 'eval'):


