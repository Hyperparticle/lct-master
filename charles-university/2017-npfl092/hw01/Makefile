t2:
	wget http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/data/skakalpes-il2.txt

t3:
	cat skakalpes-il2.txt
	less skakalpes-il2.txt

t4:
	cat skakalpes-il2.txt | iconv -f iso-8859-2 -t utf-8 > skakalpes-il2-utf8.txt

t5:
	less skakalpes-il2-utf8.txt

t6:
	cat skakalpes-il2-utf8.txt | wc -l

t7:
	cat skakalpes-il2-utf8.txt | head -n 15
	cat skakalpes-il2-utf8.txt | tail -n 15
	cat skakalpes-il2-utf8.txt | head -n 20 | tail -n 11

t8:
	cat skakalpes-il2-utf8.txt | cut -f 1,2 -d " " -s

t9:
	cat skakalpes-il2-utf8.txt | grep -e [0-9]

t10:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g'

t11:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep .

t12:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . | sort

t13:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . | sort | wc -l

t14:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . | sort | uniq | wc -l

t15:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . | sort | uniq -c | sort -nr

t16:
	cat skakalpes-il2-utf8.txt | sed 's/./&\n/g' | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . | sort | uniq -c | sort -nr

t17:
	cat skakalpes-il2-utf8.txt | sed -E -e 's/ |[[:punct:]]/\n/g' | grep . > first.txt
	cat first.txt | tail -n +2 > second.txt
	paste first.txt second.txt | sort | uniq -c | sort -nr

t18:
	bash ./bigrams.sh
