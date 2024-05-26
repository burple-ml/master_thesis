#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:38:31 2019

@author: raghav
"""
# This module augments the dataset using some of the strategies suggested in
# the unified solution paper
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

from lrraman_reader import *

# saving the original spectrum variables
np.save('ImportantVariables/Original_RRUFFspectra.npy', Ramandata_Raw_final)
np.save('ImportantVariables/Original_RRUFFMineralNames.npy', Minerals_Outputs_final)

# Showing Within class variance in the class with maximum minerals
unique_values, unique_indices, counts = np.unique(Mineral_Targets_values, 
                                                  return_index=True,
                                                  return_counts=True)
max_counts = max(counts)
max_counts_value = np.where(counts == max_counts)
Same_Class_indices = np.where(Mineral_Targets_values == max_counts_value[0])
Same_Class_spectra = Ramandata_Raw_final[Same_Class_indices]
print(Same_Class_spectra.shape)
print("The name of the mineral is : ",
      Minerals_Outputs_final[Same_Class_indices][0])
plt.figure()
plt.title('Plot explaining within class variance for a particular Mineral: '
          + Minerals_Outputs_final[Same_Class_indices][0])
for i in range(Same_Class_spectra.shape[0]):
    Same_Class_Normalised = Same_Class_spectra[i, :]/max(Same_Class_spectra[i,:])
    plt.plot(Same_Class_Normalised)
plt.savefig('ImportantPlots/Same_Class_Variance.png')
plt.show()
print("----------------------------------------------------")

# removing the spectrum classes with only 2 and 3 spectra's each 
a, counts = np.unique(Minerals_Outputs_final, return_counts=True)
b = np.where(counts == 2)
c = np.where(counts == 3)
indexes_toremove = []

for i in b[0]:
    Mineral_names_toremove = np.where(Minerals_Outputs_final == a[i])
    indexes_toremove += list(Mineral_names_toremove[0])
indexes_toremove2 = []
for i in c[0]:
    Mineral_names_toremove = np.where(Minerals_Outputs_final == a[i])
    indexes_toremove2 += list(Mineral_names_toremove[0])

# Removing those indices
Indices_delete = indexes_toremove + indexes_toremove2
Ramandata_Allspectra = np.delete(Ramandata_Raw_final, Indices_delete, axis=0)
Targets_Allspectra = np.delete(Minerals_Outputs_final, Indices_delete)
print(Ramandata_Allspectra.shape, Targets_Allspectra.shape)

np.save('ImportantVariables/Reduced_RRUFFspectra', Ramandata_Allspectra)
np.save('ImportantVariables/Reduced_RRUFFMineralNames', Targets_Allspectra)
np.save('ImportantVariables/Reduced_RRUFFMineralDict', Mineral_Dict)

# Making a dictionary and giving it values
print(len(set(Targets_Allspectra)))
Mineral_Names = sorted(set(Targets_Allspectra))
Mineral_Dict = dict(zip(Mineral_Names, range(len(set(Targets_Allspectra)))))
Mineral_Targets_values = np.array([Mineral_Dict[value] for value in Targets_Allspectra]).T

print("Spectrums removed and saved")
print("--------------------------------------------------")


# Making test validation and training split
X = Ramandata_Allspectra
y = Mineral_Targets_values
# Randomly shuffling all the data
print(Ramandata_Allspectra.shape, Mineral_Targets_values.shape)
y = y.reshape(y.shape[0], 1)
print(X.shape, y.shape)
X_ymatrix = np.append(X, y, axis=1)
np.random.shuffle(X_ymatrix)
X = X_ymatrix[:, 0:X.shape[1]]
y = X_ymatrix[:, X.shape[1]]
print(y, y.shape, X.shape)

unique_values_test, unique_indices_test, counts_test = np.unique(y, return_index=True, return_counts=True)

X_test_validation = X[unique_indices_test]
y_test_validation = y[unique_indices_test]

print(y_test_validation.shape)
print(X_test_validation.shape)

X_train = np.delete(X, unique_indices_test, axis=0)
y_train = np.delete(y, unique_indices_test)

print(X_train.shape, y_train.shape)

# Saving the dataset
np.save('ImportantVariables/Original_X_train', X_train)
np.save('ImportantVariables/Original_Y_train', y_train)
np.save('ImportantVariables/Original_X_test', X_test_validation)
np.save('ImportantVariables/Original_Y_test', y_test_validation)

print("Training and test sets made and data saved")
print("-------------------------------------------")


# Showing the histogram for the Class Counts
a, counts = np.unique(Mineral_Targets_values, return_counts=True)
hist = np.histogram(counts, )
plt.figure()
plt.title('Class Proportions')
plt.hist(counts, np.arange(40))
plt.xlabel('The count of each class')
plt.ylabel('Number of such Classes')
plt.savefig('ImportantPlots/ClassProportions.png')
plt.show()

print("Class Proportions shown and figure saved")
print("-------------------------------------------")


# Data Augmentation

print("Time for some Data Augmentation")
print("The original shapes and types are as follows")
print(Ramandata_Raw_final.shape, Minerals_Outputs_final.shape)
print(Ramandata_Allspectra.shape, Mineral_Targets_values.shape)
print(X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)

# print(augmented_targets.shape,augmented_targets)
augmented_spectra = np.ones((1,701), dtype='float64')
augmented_targets = np.array((1,), dtype='float64')
for j in range(3):
    d1 = np.arange(-15, 15, 1)
    d = np.delete(d1, np.where(d1 == 0))
    print(d)
    for i in range(X_train.shape[0]):
        spectrum1 = X_train[i]
        coin_toss = np.random.choice(d)
        spectrum2 = np.roll(spectrum1, coin_toss)
        if np.sign(coin_toss) < 0:
            spectrum3 = spectrum2[:coin_toss]
            spectrum4 = np.pad(spectrum3, (0, np.absolute(coin_toss)), 'edge')
        else:
            spectrum3 = spectrum2[coin_toss:]
            spectrum4 = np.pad(spectrum3, (coin_toss, 0), 'edge')
        spectrum4 = spectrum4.reshape(1, 701)
        augmented_spectra = np.append(augmented_spectra, spectrum4, axis=0)
        augmented_targets = np.append(augmented_targets, y_train[i])
augmented_spectra = augmented_spectra[1:, :]
augmented_targets = augmented_targets[1:]
X_train_augmented1 = np.append(X_train, augmented_spectra, axis=0)
y_train_augmented1 = np.append(y_train, augmented_targets)

plt.figure()
plt.title('Observe the shifted spectrums')
for i in range(5):
    plt.plot(augmented_spectra[i, :]/max(augmented_spectra[i, :]))
    plt.plot(X_train[i, :]/max(X_train[i, :]))
plt.savefig('ImportantPlots/Data_Augmentation1.png')
plt.show()

print("The shape after first augmentation are as follows")
print(X_train_augmented1.shape, y_train_augmented1.shape)
print("-------")


# Data augmentation - added noise proportional to magnitude at each wavenumber
augmented_spectra = np.ones((1, 701), dtype='float64')
augmented_targets = np.array((1,), dtype='float64')
for j in range(3):
    for i in range(X_train.shape[0]):
        d = [10, 25, 50, 75, 100]
        coin_toss = np.random.choice(d)
        a = X_train[i]
        b = y_train[i]
        a = a + np.random.normal(loc=0, scale=np.absolute(a)/coin_toss)
        # added noise here
        a = a.reshape(1, 701)
        augmented_spectra = np.append(augmented_spectra, a, axis=0)
        augmented_targets = np.append(augmented_targets, b)
augmented_spectra = augmented_spectra[1:, :]
augmented_targets = augmented_targets[1:]
X_train_augmented2 = np.append(X_train_augmented1, augmented_spectra, axis=0)
y_train_augmented2 = np.append(y_train_augmented1, augmented_targets)

plt.figure()
plt.title('Spectra with added noise')
for i in range(5):
    plt.plot(augmented_spectra[i, :]/max(augmented_spectra[i]))
    plt.plot(X_train[i, :]/max(X_train[i]))
plt.savefig('ImportantPlots/Data_Augmentation2.png')
plt.show()

print("The shapes after data augmentation are as follows")
print(X_train_augmented2.shape, y_train_augmented2.shape)
print("------")

# Randomly shuffling all the training data
print(X_train_augmented2.shape, y_train_augmented2.shape)
y_train_augmented2 = y_train_augmented2.reshape(y_train_augmented2.shape[0], 1)
print(X_train_augmented2.shape, y_train_augmented2.shape)
X_ytrainmatrix = np.append(X_train_augmented2, y_train_augmented2, axis=1)
np.random.shuffle(X_ytrainmatrix)
X_train_final = X_ytrainmatrix[:, 0:X_train_augmented2.shape[1]]
y_train_final = X_ytrainmatrix[:, X_train_augmented2.shape[1]]
print(y_train_final, y_train_final.shape, X_train_final.shape)

print("DATA IS SHUFFLED AND AUGMENTED")
np.save('ImportantVariables/X_train_final_unnormalised', X_train_final)
np.save('ImportantVariables/y_train_final_unnormalised', y_train_final)
print("------------------------------------------")

# Normalising the spectra to max value
print(X_train_final.shape)
# plt.plot(X_train_final[0,:])
# plt.show()
max_values = np.amax(X_train_final, axis=1)
print(max_values.shape)
X_train_final = X_train_final/max_values.reshape(X_train_final.shape[0], 1)
# plt.plot(X_train_final[0,:])
# plt.show()

# normalising by max value for the X_test_validation
print(X_test_validation.shape)
# plt.plot(X_test_validation[0,:])
# plt.show()
max_values = np.amax(X_test_validation, axis=1)
print(max_values.shape)
X_test_validation = X_test_validation/max_values.reshape(X_test_validation.shape[0],1)
# plt.plot(X_test_validation[0,:])
# plt.show()
print(X_train_final.shape, X_test_validation.shape)
print("DATA IS NORMALISED AND READY FOR TRAINING")
np.save('ImportantVariables/X_train_final_normalised', X_train_final)
np.save('ImportantVariables/y_train_final_normalised', y_train_final)
np.save('ImportantVariables/X_test_validation_normalised', X_test_validation)
np.save('ImportantVariables/y_test_validation_normalised', y_test_validation)
print("DATA IS ALSO SAVED")
print("-------------------------------------------------")
