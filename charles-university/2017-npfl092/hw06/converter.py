#!/usr/bin/env python3

import xml.etree.ElementTree as ET

tsvfile = 'dependency_trees_from_ud.tsv'
xmlfile = 'dependency_trees_from_ud.xml'

with open(tsvfile) as f:
  for line in f:
    split = line.split()
    print(split)

# tree = ET.ElementTree()
# root = tree.getroot()