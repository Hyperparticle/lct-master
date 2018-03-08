all: download perc knn nb dt

download:
	cd data; wget http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

	cd data; tar -xzf artificial_objects.tgz

perc:
	python ./perceptron.py

knn:
	python ./k-nearest-neighbors.py

nb:
	python ./naive-bayes.py

dt:
	python ./decision-trees.py
