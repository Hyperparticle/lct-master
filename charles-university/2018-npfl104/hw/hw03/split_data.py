#!/usr/bin/env python3

import pandas as pd
import numpy as np
import pdb
from sklearn.model_selection import train_test_split

def dump(xydata, split="train"):
  data_xout = open("x_%s.csv" % split,'w')
  data_yout = open("y_%s.csv" % split,'w')

  n,m = xydata.shape
  
  for i in range(n):
    try:
      data_xout.write(",".join([str(x) for x in xydata[i,:-1]] ) + '\n')
    except:
      pdb.set_trace()
    data_yout.write(str(xydata[i,-1]) + '\n')

  data_xout.close()
  data_yout.close()


if __name__ == "__main__":
  data = pd.read_csv("creditcard.csv")

  Y = data["Class"]
  xc = data[data.columns.difference(["Class"])]
  X = xc.as_matrix()
  c1_idx = Y==1

  xt1,xts1,yt1,yts1 = train_test_split(X[c1_idx,:],Y[c1_idx],test_size=.2,random_state=42)
  xt2,xts2,yt2,yts2 = train_test_split(X[~c1_idx,:],Y[~c1_idx],test_size=.2,random_state=42)

  xtrain = np.vstack([xt1,xt2])
  ytrain = np.hstack([yt1,yt2]).reshape([-1,1])
  xtest = np.vstack([xts1,xts2])
  ytest = np.hstack([yts1,yts2]).reshape([-1,1])

  xytrain = np.hstack([xtrain,ytrain])
  xytest = np.hstack([xtest,ytest])
  np.random.shuffle(xytrain)
  np.random.shuffle(xytest)

  dump(xytrain,"train")
  dump(xytest,"test")