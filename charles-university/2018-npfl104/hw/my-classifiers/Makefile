all: download perc nb knn dt

download:
	echo "" > results.txt
	cd data; wget http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl104/html/classification_data/artificial_objects.tgz
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
	cd data; wget http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test

	cd data; tar -xzf artificial_objects.tgz

perc:
	echo "perc" | tee -a results.txt
	python3 ./perceptron.py | tee -a results.txt

knn:
	echo "knn" | tee -a results.txt
	python3 ./k-nearest-neighbors.py | tee -a results.txt

nb:
	echo "nb" | tee -a results.txt
	python3 ./naive-bayes.py | tee -a results.txt

dt:
	echo "dt" | tee -a results.txt
	python3 ./decision-trees.py | tee -a results.txt
