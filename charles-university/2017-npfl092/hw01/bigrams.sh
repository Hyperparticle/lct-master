#!/bin/bash

wget http://textfiles.com/news -O story.txt

echo '------------'
echo 'Bigrams:'
cat story.txt | sed 's/<[^>]\+>//g' | sed -E -e 's/\s|[[:punct:]]/\n/g' | grep -E -e '[A-Z][a-z]*' > story-first.txt
cat story-first.txt | tail -n +2 > story-second.txt
paste story-first.txt story-second.txt | sort | uniq -c | sort -nr | head

echo '------------'
echo 'HTML tags:'
cat story.txt | grep -o '<\(/\)\?[A-Za-z0-9]\+' | grep -o '[A-Za-z0-9]\+' | grep . | sort | uniq -c | sort -nr
