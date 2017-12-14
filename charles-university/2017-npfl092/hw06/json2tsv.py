#!/usr/bin/env python3

import json

jsonfile = 'dependency_trees_from_ud.json'
tsvfile = 'dependency_trees_from_ud_gen.tsv'

with open(jsonfile) as j:
  data = json.load(j)

lines = []
for sentence in data['data']:
  lines.append(sentence['comment'].strip())
  compress = lambda word: [word['id'], word['token'], word['stem'], word['tag'], word['dependency']]
  for word in sentence['words']:
    lines.append('\t'.join(compress(word)))
  lines.append('')

lines.append('')

with open(tsvfile, 'w') as tsv:
  tsv.write('\n'.join(lines))
