#!/usr/bin/env python3

import pandas as pd
import numpy as np

if __name__ == "__main__":
  data = pd.read_csv('creditcard.csv')[['V' + str(i+1) for i in range(28)] + ['Class']]

  true = data[data.Class == 1]
  false = data[data.Class == 0][:5000 - len(true)]

  s0, s1 = int(len(true) * 0.8), int(len(false) * 0.8) 
  true_train, true_test = true[:s0], true[s0:]
  false_train, false_test = false[:s1], false[s1:]

  train = true_train.append(false_train).sample(frac=1).reset_index(drop=True)
  test = true_test.append(false_test).sample(frac=1).reset_index(drop=True)
  
  train.to_csv('train.txt', header=False, index=False)
  test.to_csv('test.txt', header=False, index=False)
 