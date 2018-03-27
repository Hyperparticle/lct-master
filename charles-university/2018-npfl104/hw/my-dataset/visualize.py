#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_csv('creditcard.csv')

x,y = data['V1'], data['V2']

plt.scatter(x, y)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('V2 as a function of V1')

# slope,intercept = curve_fit(lambda x,a,b: a*x + b, x, y)[0]
# print(slope, intercept)
# plt.plot([70, 90], [90, 200], 'k-')

plt.savefig('data.png')
