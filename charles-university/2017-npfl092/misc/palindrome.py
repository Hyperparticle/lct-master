#!/usr/bin/env python3

# reads text from stdin and prints detected palindromes (one per line) to stdout, print only palindrome words longer than three letters

import fileinput

def is_palindrome(text):
    if (len(text) <= 3):
        return False
    return all(text[i] == text[-i-1] for i in range(len(text) // 2))

for line in fileinput.input():
    tokens = line.split()
    palindromes = (token for token in tokens if is_palindrome(token))

    for palindrome in palindromes:
        print(palindrome)
