#!/usr/bin/env python3

# 1
genesis = "In the beginning God created the heavens and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light. And God saw the light, that it was good: and God divided the light from the darkness. And God called the light Day, and the darkness he called Night. And the evening and the morning were the first day. And God said, Let there be a firmament in the midst of the waters, and let it divide the waters from the waters. And God made the firmament, and divided the waters which were under the firmament from the waters which were above the firmament: and it was so. And God called the firmament Heaven. And the evening and the morning were the second day. And God said, Let the waters under the heaven be gathered together unto one place, and let the dry land appear: and it was so. And God called the dry land Earth; and the gathering together of the waters called he Seas: and God saw that it was good. And God said, Let the earth bring forth grass, the herb yielding seed, and the fruit tree yielding fruit after his kind, whose seed is in itself, upon the earth: and it was so. And the earth brought forth grass, and herb yielding seed after his kind, and the tree yielding fruit, whose seed was in itself, after his kind: and God saw that it was good. And the evening and the morning were the third day. And God said, Let there be lights in the firmament of the heaven to divide the day from the night; and let them be for signs, and for seasons, and for days, and years: And let them be for lights in the firmament of the heaven to give light upon the earth: and it was so. And God made two great lights; the greater light to rule the day, and the lesser light to rule the night: he made the stars also. And God set them in the firmament of the heaven to give light upon the earth, And to rule over the day and over the night, and to divide the light from the darkness: and God saw that it was good. And the evening and the morning were the fourth day. And God said, Let the waters bring forth abundantly the moving creature that hath life, and fowl that may fly above the earth in the open firmament of heaven. And God created great whales, and every living creature that moveth, which the waters brought forth abundantly, after their kind, and every winged fowl after his kind: and God saw that it was good. And God blessed them, saying, Be fruitful, and multiply, and fill the waters in the seas, and let fowl multiply in the earth. And the evening and the morning were the fifth day. And God said, Let the earth bring forth the living creature after his kind, cattle, and creeping thing, and beast of the earth after his kind: and it was so. And God made the beast of the earth after his kind, and cattle after their kind, and every thing that creepeth upon the earth after his kind: and God saw that it was good. And God said, Let us make man in our image, after our likeness: and let them have dominion over the fish of the sea, and over the fowl of the air, and over the cattle, and over all the earth, and over every creeping thing that creepeth upon the earth. So God created man in his own image, in the image of God created he him; male and female created he them. And God blessed them, and God said unto them, Be fruitful, and multiply, and replenish the earth, and subdue it: and have dominion over the fish of the sea, and over the fowl of the air, and over every living thing that moveth upon the earth. And God said, Behold, I have given you every herb bearing seed, which is upon the face of all the earth, and every tree, in the which is the fruit of a tree yielding seed; to you it shall be for meat. And to every beast of the earth, and to every fowl of the air, and to every thing that creepeth upon the earth, wherein there is life, I have given every green herb for meat: and it was so. And God saw every thing that he had made, and, behold, it was very good. And the evening and the morning were the sixth day."
print(genesis[:40])
print(genesis[3:6])
print(len(genesis))

# 2
tokens = genesis.split()
print(tokens[:10])
print(tokens[-10:])
print(tokens[10:18])

# 3
from collections import Counter
counts = Counter(tokens)

# 4
print(counts.most_common()[0])

# 5
print(counts.most_common()[:10])

# 6 Get unigrams with count > 5
print([token for token in counts.keys() if counts[token] > 5])

# 7 Count bigrams in the text into a dict of Counters
from collections import defaultdict
bigrams = defaultdict(Counter)
for wprev,w in zip(tokens, tokens[1:]):
  bigrams[wprev][w] +=1

# 8 For each unigram with count > 5, print to together with its most frequent successor.
print([(w, bigrams[w].most_common()[0][0]) for w in counts.keys() if counts[w] > 5])

# 9 Print the successor together with its relative frequency rounded to 2 decimal digits.
print([(bigrams[w].most_common()[0][0], round(bigrams[w].most_common()[0][1] / len(tokens), 2)) for w in counts.keys() if counts[w] > 5])

# 10 Print a random token. Print a random unigram disregarding their distribution.
import random
print(random.choice(list(counts.keys())))

# 11 Pick a random word, generate a string of 20 words by always picking the most frequent follower.
words = [random.choice(list(counts.keys()))]
for i in range(20):
  words.append(bigrams[words[-1]].most_common()[0][0])
print(words)

# 12 Put that into a function, with the number of words to be generated as a parameter. Return the result in a list.
def most_freq_str():
  words = [random.choice(list(counts.keys()))]
  for i in range(20):
    words.append(bigrams[words[-1]].most_common()[0][0])
  return words

# 13 Sample the next word according to the bigram distribution
import numpy as np
def most_freq_seq(n):
  words = [random.choice(list(counts.keys()))]
  for i in range(n):
    word = words[-1]
    next_words = list(bigrams[word].keys())
    probs = [bigrams[word][n] / counts[word] for n in next_words]
    words.append(np.random.choice(next_words, p=probs))
  return words

print(most_freq_seq(20))
