#!/usr/bin/env python3

import xml.etree.ElementTree as ET

xmlfile = 'dependency_trees_from_ud.xml'
jsonfile = 'dependency_trees_from_ud.json'

with open(xmlfile) as xml:
  with open(jsonfile, 'wb') as j:
    
