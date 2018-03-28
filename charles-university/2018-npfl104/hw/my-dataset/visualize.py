#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv('creditcard.csv')

true = data[data.Class == 1]
false = data[data.Class == 0]

x = [true.Amount, false.Amount]

# plt.hist(true.Amount, 100, range=(0, 1000), log=True)
plt.hist(x, 100, range=(0, 1500), log=True, label=['fraud', 'normal'], color=['tab:red', 'tab:blue'], histtype='bar', stacked=True)

plt.legend(prop={'size': 14})

plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.title('Frequency of Purchases by Value')

plt.savefig('data.png')
