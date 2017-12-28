#!/usr/bin/env python3

# Calculates statistics across all data points

import sys
from collections import defaultdict
from collections import Counter
import numpy as np

counts = defaultdict(lambda: [])

for line in sys.stdin:
    s = line.split()
    if len(s) == 3:
        m, t, steps = int(s[0]), int(s[1]), int(s[2])
        if len(counts[m]) <= t:
            counts[m].append([])
        counts[m][t].append(steps)
    else:
        print(line)

print('m', 'min', 'max', 'avg', 'median', 'decile', sep='\t')

for m in sorted(counts):
    stats = np.average(counts[m], axis=1)

    min_stats = np.min(stats)
    max_stats = np.max(stats)
    avg_stats = np.average(stats)
    med_stats = np.median(stats)
    decile_stats = np.percentile(stats, 90)

    print(m, min_stats, max_stats, avg_stats, med_stats, decile_stats, sep='\t')
