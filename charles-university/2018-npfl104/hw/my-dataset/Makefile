all: download prepare split visualize show

download:
	# wget https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/2
	wget http://www3.zippyshare.com/d/CWGYJKV0/68924/creditcard.tar.gz \
	--referer='http://www3.zippyshare.com/v/CWGYJKV0/file.html' \
	--cookies=off --header "Cookie: JSESSIONID=07555DEF1A8B7DB812C30ECFC4C59424" \
	--user-agent='Mozilla/5.0 (Windows NT 6.0) Gecko/20100101 Firefox/14.0.1' \
	-O creditcard.tar.gz
	tar -xzvf creditcard.tar.gz

prepare:
	cat creditcard.csv | grep -v "^\"" > data.csv

split:
	cat data.csv | head -n  230000 > train.txt
	cat data.csv | tail -n +230001 > test.txt

visualize:
	python3 visualize.py

show:
	feh data.png
