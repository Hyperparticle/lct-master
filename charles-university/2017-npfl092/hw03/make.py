#!/usr/bin/env python3

import sys
import urllib.request
import regex as re
from collections import Counter

def t2():
    response = urllib.request.urlopen('http://ufal.mff.cuni.cz/~zabokrtsky/courses/npfl092/html/data/skakalpes-il2.txt')
    content = response.read()
    with open('skakalpes-il2.txt', 'wb') as f:
        f.write(content)

def t3():
    with open('skakalpes-il2.txt', 'rb') as f:
        print(f.read())

def t4():
    with open('skakalpes-il2.txt', 'rb') as f:
        content = f.read().decode('iso-8859-2')
    
    with open('skakalpes-il2-utf8.txt', 'w') as f:
        f.write(content)

def t5():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        print(f.read())

def t6():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        print(sum(1 for line in f))

def t7():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = list(f)
        print(''.join(lines[:15]).strip())
        print(''.join(lines[-15:]).strip())
        print(''.join(lines[9:20]).strip())

def t8():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        for line in f:
            words = line.split()
            if (len(words) != 0):
                print(' '.join(words[:2]))

def t9():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        for line in f:
            if (re.search(r'\d', line)):
                print(line.strip())

def t10():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        for line in f:
            print(re.sub(r' |[[:punct:]]', r'\n', line.strip()))

def t11():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0)
        print('\n'.join(words))

def t12():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0)
        sort = sorted(words)
        print('\n'.join(sort))

def t13():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0)
        sort = sorted(words)
        print(sum(1 for line in sort))

def t14():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0)
        sort = sorted(words)
        vocab = set(sort)
        print(sum(1 for line in vocab))

def t15():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0)
        counts = Counter(words)
        for word,freq in counts.most_common():
            print(str(freq).rjust(7) + ' ' + word)

def t16():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        chars = (c.strip() for line in lines for c in line)
        processed = (c for c in chars if len(c) != 0 and not re.search(r'[[:punct:]]', c))
        counts = Counter(processed)
        for c,freq in counts.most_common():
            print(str(freq).rjust(7) + ' ' + c)

def t17():
    with open('skakalpes-il2-utf8.txt', 'r') as f:
        lines = (line.strip() for line in f)
        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = [word for s in sentences for word in s if len(word.strip()) != 0]
        bigrams = zip(words, words[1:])
        counts = Counter(bigrams)
        for bigram,freq in counts.most_common():
            print(str(freq).rjust(7) + ' ' + ' '.join(bigram))

def t18():
    response = urllib.request.urlopen('http://textfiles.com/news')
    content = response.read().decode('utf-8')
    with open('story.txt', 'w') as f:
        f.write(content)

    bigrams = []
    tag_counts = Counter()

    with open('story.txt', 'r') as f:
        lines = [line.strip() for line in f]

        tags = [tag[1:] for line in lines for tag in re.findall('<[a-zA-Z0-9]+', line)]
        tag_counts.update(tags)

        sentences = (re.split(r' |[[:punct:]]', line) for line in lines)
        words = (word for s in sentences for word in s if len(word.strip()) != 0 and word not in tag_counts.keys())
        capitals = [word for word in words if re.search('^[[:upper:]][[:alpha:]]*', word)]
        bigrams = list(zip(capitals, capitals[1:]))
        

    print('------------')
    print('Bigrams:')
    for bigram in bigrams:
        print(' '.join(bigram))


    print('------------')
    print('HTML Tags:')

    for tag,freq in tag_counts.most_common():
        print(str(freq).rjust(7) + ' ' + tag)
    

if len(sys.argv) < 2:
    # No arguments -> all exercises
    exercises = ['t' + str(i) for i in range(2, 19)]
    for exercise in exercises:
        locals()[exercise]()
elif sys.argv[1] in locals():
    # Run the appropriate exercise
    locals()[sys.argv[1]]()
else:
    print(sys.argv[1] + ' is not a valid exercise')
