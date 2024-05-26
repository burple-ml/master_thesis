#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 07:43:40 2019

@author: raghav
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

directory = os.getcwd() + '/dataset'

# dtypes managed later in the python notebook.
ramandata_raw = np.ones((1, 1001), dtype='object')
diabetes = np.ones((1, ), dtype='object') 

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        #reading raw file using pandas
        print(filename)
        rawfile = np.asarray(pd.read_csv(filename))
        
        ramandata_raw = np.append(ramandata_raw, rawfile[:,801:1802],axis=0)
        diabetes = np.append(diabetes, rawfile[0:,0])

raman_shift = np.arange(800,1801)

plt.title('Observe the variance amongs different kinds of signals')
plt.plot(raman_shift, ramandata_raw[np.random.randint(30)], 'r',
         raman_shift, ramandata_raw[np.random.randint(30)], 'b',
         raman_shift, ramandata_raw[np.random.randint(30)],'g')
plt.show()
plt.plot(raman_shift, ramandata_raw[np.random.randint(30)], 'r',
         raman_shift, ramandata_raw[np.random.randint(30)],'b', 
         raman_shift, ramandata_raw[np.random.randint(30)],'g')
plt.show()

# randomly shuffling the signals to balance cross validation partitions
diabetes = diabetes.reshape(80,1)
shufflematrix = np.append(ramandata_raw, diabetes, axis=1)
ramandata = np.take(shufflematrix, np.random.permutation(shufflematrix.shape[0]), axis=0)

ramandata_raw_final = ramandata[:, :1001]
diabetes = ramandata[:, 1001]