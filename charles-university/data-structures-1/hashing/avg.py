#!/usr/bin/env python3

import sys

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

lines = sys.stdin.readlines()

for chunk in split(lines, 100):
    sum1, sum2, sum3 = 0, 0, 0
    for line in chunk:
        s = line.split()
        if (len(s) == 3):
            steps, alpha, time = s
            sum1 += float(steps)
            sum2 += float(alpha)
            sum3 += float(time)
        else:
            print(line)
    print(sum1 / len(chunk), sum2 / len(chunk), sum3 / len(chunk))
