all: download prepare split visualize show

download:
	wget https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/2

prepare:
	cat creditcard.csv | grep -v "^\"" > data.csv

split:
	cat data.csv | head -n  230000 > train.txt
	cat data.csv | tail -n +230001 > test.txt

visualize:
	python3 visualize.py

show:
	eog data.png
