all: download tsv2xml xml2json json2tsv check

download:
	wget https://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/data/dependency_trees_from_ud.tsv

tsv2xml:
	./tsv2xml.py

xml2json:
	./xml2json.py

json2tsv:
	./json2tsv.py

check:
	diff ./dependency_trees_from_ud.tsv ./dependency_trees_from_ud_gen.tsv