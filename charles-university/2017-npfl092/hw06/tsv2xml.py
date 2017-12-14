#!/usr/bin/env python3

import xml.etree.ElementTree as ET

tsvfile = 'dependency_trees_from_ud.tsv'
xmlfile = 'dependency_trees_from_ud.xml'

with open(tsvfile) as tsv:
  with open(xmlfile, 'wb') as xml:
    data = ET.Element('data')
    doc = ET.ElementTree(data)

    for line in tsv:
      if line.startswith('#'):
        sentence = ET.SubElement(data, 'sentence')
        sentence.set('comment', line)
      elif line.strip():
        id, token, stem, tag, dependency = line.strip().split('\t')

        word = ET.SubElement(sentence, 'word')

        word.text = token

        word.set('id', id)
        word.set('stem', stem)
        word.set('dependency', dependency)
        word.set('tag', tag)

    doc.write(xml)