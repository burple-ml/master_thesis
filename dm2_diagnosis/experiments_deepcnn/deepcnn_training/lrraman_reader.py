#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:31:51 2019

@author: raghav
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt


# Specify the directory and file name here:-
directory = os.path.join(os.getcwd(), 'dataset/minerals')

# Create data array to load
Ramandata_Raw1 = np.ones((1, 701), dtype=np.float64)
Minerals_output = np.array(['xyz'], dtype='object')

# Read file using loadtxt
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filename = os.path.join(directory, filename)
        Ramandata = np.loadtxt(filename, delimiter=',', comments='#')
        if (Ramandata[0, 0] - 150 > 1.9285):
            num_zeros = int((Ramandata[0, 0] - 150)/1.9285)
            # zeros_bufferarray=np.zeros(num_zeros)
            num_tobeclipped = 701-num_zeros
            Clipped_data3 = Ramandata[0:num_tobeclipped, 1]
            # Clipped_data4=np.append(zeros_bufferarray,Clipped_data3)
            if Clipped_data3.shape[0]+num_zeros < 701:
                N = 701-Clipped_data3.shape[0]-num_zeros
                Clipped_data4 = np.pad(Clipped_data3, (num_zeros, N), 'edge')
            else:
                Clipped_data4 = np.pad(Clipped_data3, (num_zeros, 0), 'edge')
            Clipped_data4 = Clipped_data4.reshape(1, 701)
            Ramandata_Raw1 = np.append(Ramandata_Raw1, Clipped_data4, axis=0)
            file = open(filename, 'r')
            name_mineral = file.readline()
            name_mineral1 = name_mineral[8:]
            Minerals_output = np.append(Minerals_output, name_mineral1)
            file.close()
        else:
            threshold_value = np.where(Ramandata[:, 0] > 150)
            Clipped_data = Ramandata[threshold_value]
            Clipped_data2 = Clipped_data[0:701, 1]
            if Clipped_data2.shape[0] < 701:
                N = 701-Clipped_data2.shape[0]
                Clipped_data2 = np.pad(Clipped_data2, (0, N), 'edge')
            Clipped_data2 = Clipped_data2.reshape(1, 701)
            Ramandata_Raw1 = np.append(Ramandata_Raw1, Clipped_data2, axis=0)
            file = open(filename, 'r')
            name_mineral = file.readline()
            name_mineral1 = name_mineral[8:]
            Minerals_output = np.append(Minerals_output, name_mineral1)
            file.close()

Raman_shifts = np.arange(150, 1500, 1.9285)
print("Shape of Raman Shifts-->", Raman_shifts.shape,
      "First 2 and Last Raman shifts--->",
      Raman_shifts[0], Raman_shifts[1], Raman_shifts[700])

plt.figure()
plt.title('A sample spectrum collected from RRUFF dataset: '+Minerals_output[1])
plt.plot(Raman_shifts, Ramandata_Raw1[1])
plt.xlabel('Raman_Shifts')
plt.ylabel('Intensities')
plt.savefig('ImportantPlots/sample_RRUFF_spectrum.png')
plt.show()

Ramandata_Raw = Ramandata_Raw1[1:]
Minerals_Outputs = Minerals_output[1:]

# Removing singular values
u, indices, counts_unique = np.unique(Minerals_Outputs, return_index=True,
                                      return_counts=True)
# checking singular values
abc = np.where(counts_unique == 1)
Minerals_Outputs_final = np.delete(Minerals_Outputs, indices[abc])
Ramandata_Raw_final = np.delete(Ramandata_Raw, indices[abc], axis=0)

# Printing the some mineral names as checkpoint
print(len(set(Minerals_Outputs_final)))
Mineral_Names = sorted(set(Minerals_Outputs_final))
Mineral_Dict = dict(zip(Mineral_Names, range(len(set(Minerals_Outputs_final)))))
Mineral_Targets_values = np.array([Mineral_Dict[value] for value in Minerals_Outputs_final]).T
print("everything upto here is done", Mineral_Targets_values.shape)
print("The shape of Ramandata_Raw_final is:", Ramandata_Raw_final.shape)
# Mineral_Target_values=np.asarray(Mineral_Targets)
