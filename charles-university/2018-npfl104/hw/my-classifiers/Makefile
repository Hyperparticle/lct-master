all: download perc nb knn dt

download:
	cd data; wget http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

	cd data; tar -xzf artificial_objects.tgz

perc:
	python3 ./perceptron.py

knn:
	python3 ./k-nearest-neighbors.py

nb:
	python3 ./naive-bayes.py

dt:
	python3 ./decision-trees.py
