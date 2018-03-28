#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv('creditcard.csv')

x,y = data['Amount'], data['Class']

plt.scatter(x, y)
plt.xlabel('Amount')
plt.ylabel('Class')
# plt.title('V2 as a function of V1')

plt.savefig('data.png')
