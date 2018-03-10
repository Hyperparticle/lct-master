#!/usr/bin/env python3
# coding=utf-8

# 10 tasks from Torbjörn Lager's 46 simple Python Exercices 
# https://github.com/PabloVallejo/python-exercises/blob/master/exercises-list.md

# 1. Define a function max() that takes two numbers as arguments and returns the largest of them.
def max(a, b):
    if (a > b):
        return a
    else:
        return b

# 2. Define a function max_of_three() that takes three numbers as arguments and returns the largest of them.
def max_of_three(a, b, c):
    return max(max(a, b), c)

# 3. Define a function that computes the length of a given list or string.
def length(x):
    count = 0
    for _ in x:
        count += 1
    return count

# 4. Write a function that takes a character ( i.e. a string of length 1 ) and returns True if it is a vowel, False otherwise.
def vowel(s):
    return s in 'aeiou'

# 5. Write a function translate() that will translate a text into "rövarspråket" (Swedish for "robber's language"). That is, double every consonant and place an occurrence of "o" in between.
def translate(s):
    return ''.join(c+'o'+c.lower() if c.lower() in 'bcdfghjklmnpqrstvwxyz' else c for c in s)
    
# 6. Define a function sum() and a function multiply() that sums and multiplies (respectively) all the numbers in a list of numbers.
def sum(l):
    s = 0
    for i in l:
        s += i
    return s

def multiply(l):
    m = 1
    for i in l:
        m *= i
    return m

# 7. Define a function reverse() that computes the reversal of a string.
def reverse(s):
    return s[::-1]

# 8. Define a function is_palindrome() that recognizes palindromes.
def is_palindrome(s):
    return s == reverse(s)

# 9. Write a function is_member() that takes a value ( i.e. a number, string, etc ) x and a list of values a, and returns True if x is a member of a, False otherwise.
def is_member(x, a):
    return any(x == i for i in a)

# 10. Define a function overlapping() that takes two lists and returns True if they have at least one member in common, False otherwise.
def overlapping(a, b):
    return any(x == y for x in a for y in b)

assert(max(1, 2) == 2)
assert(max(2, 1) == 2)
print('max passed')

assert(max_of_three(1, 2, 3) == 3)
assert(max_of_three(1, 3, 2) == 3)
assert(max_of_three(3, 1, 2) == 3)
print('max_of_three passed')

assert(length([1, 2, 3]) == 3)
assert(length('abc') == 3)
print('length passed')

assert(vowel('s') == False)
assert(vowel('u') == True)
print('vowel passed')

assert(translate('This is fun') == 'Tothohisos isos fofunon')
print('translate passed')

assert(sum([1, 2, 3]) == 6)
assert(multiply([1, 2, 2]) == 4)
print('sum and multiply passed')

assert(reverse('abc') == 'cba')
print('reverse passed')

assert(is_palindrome('aabaa') == True)
assert(is_palindrome('abba') == True)
assert(is_palindrome('abca') == False)
print('is_palindrome passed')

assert(is_member(1, [3, 2, 1, 3]) == True)
assert(is_member(4, [3, 2, 1, 3]) == False)
print('is_member passed')

assert(overlapping([1, 2, 3], [6, 1, 4]) == True)
assert(overlapping([1, 2, 3], [4, 5, 6]) == False)
print('overlapping passed')

print('All tests passed.')
