all: download run

download:
	rm -rf 2018-npfl104-shared
	git clone ssh://gitolite@redmine.ms.mff.cuni.cz/undergrads/student/2018-npfl104-shared.git

run:
	python3 scikit-class.py | tee results.txt