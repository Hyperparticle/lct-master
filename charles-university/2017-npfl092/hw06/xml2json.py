#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import json

xmlfile = 'dependency_trees_from_ud.xml'
jsonfile = 'dependency_trees_from_ud.json'

data = ET.parse(xmlfile).getroot()

sentences = [{
  **sentence.attrib,
  'words': [{
    **w.attrib, 
    'token': w.text
  } for w in sentence]
} for sentence in data]

with open(jsonfile, 'w') as f:
  f.write(json.dumps({'data': sentences}, sort_keys=True, indent=2))
