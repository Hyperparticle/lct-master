#!/usr/bin/env python3

import sys
from collections import defaultdict
from collections import Counter
import numpy as np

counts = defaultdict(lambda: [])

for line in sys.stdin:
    s = line.split()
    if (len(s) == 2):
        m, steps = int(s[0]), int(s[1])
        counts[m].append(steps)
    else:
        print(line)

print('m', 'min', 'max', 'avg', 'median', 'decile', sep='\t')

for m in sorted(counts):
    min_stats = np.min(counts[m])
    max_stats = np.max(counts[m])
    avg_stats = np.average(counts[m])
    med_stats = np.median(counts[m])
    decile_stats = np.percentile(counts[m], 90)

    print(m, min_stats, max_stats, avg_stats, med_stats, decile_stats, sep='\t')
