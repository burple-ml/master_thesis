#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:47:53 2019

@author: raghav
"""

from ...reading_raman import *

import torch
import torch.nn as nn

from torch.nn import Linear, BatchNorm1d, Conv1d
from torch.nn.functional import relu, softmax

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV

# Dataset definition and setting a few hyperparameters
X = ramandata_raw_final.reshape(-1, 1, 1001).astype('float32')
y = diabetes.astype('int64')
print(X.shape, y.shape)

num_classes = 2
channels = X.shape[1]
height = X.shape[2]
padding_conv1 = 0


def compute_conv_dim(dim_size, k, p, s):
    """
    This function computes the size of the intermediate layers after the
    convolution operation.
    """
    return int((dim_size - k + 2 * p) / s + 1)


def compute_pool_dim(dim_size, kp, pp, sp):
    """
    This functino computes the size of the intermediate layer after the
    pooling operation.
    """
    return int((dim_size - kp + 2 * pp) / sp + 1)


class Net(nn.Module):
    """
    The neural network class definition. Inherits the pytorch class Module
    """

    def __init__(self, num_units, kernel_size_conv1, stride_conv1):
        super(Net, self).__init__()
        self.conv_1 = Conv1d(in_channels=channels,
                             out_channels=num_units,
                             kernel_size=kernel_size_conv1,
                             padding=padding_conv1,
                             stride=stride_conv1)
        self.conv_1_bn = BatchNorm1d(num_units)
        self.conv_out_height = compute_conv_dim(height, kernel_size_conv1,
                                                padding_conv1,
                                                stride_conv1)
        self.l1_in_features = num_units * self.conv_out_height
        self.l_out = Linear(in_features=self.l1_in_features,
                            out_features=num_classes,
                            bias=True)

    def forward(self,  x):
        """
        Defining the forward pass here
        """
        x = relu(self.conv_1_bn(self.conv_1(x)))
        x = x.view(-1, self.l1_in_features)
        return softmax(self.l_out(x), dim=1)


# Creating a instance of the NeuralNetClassifier class of skorch.
net = NeuralNetClassifier(
    Net,
    max_epochs=200,
    module__num_units=2,
    module__kernel_size_conv1=10,
    module__stride_conv1=1,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer__lr=0.001,
    optimizer__weight_decay=0.1,
    batch_size=8,
    # iterator_train__batch_size=16,
    # iterator_valid__batch_size=8,
    callbacks=[
        EarlyStopping(monitor='train_loss', patience=20, threshold=0.00001)
    ],
    train_split=None,
    verbose=0,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True
)

print(net)  # Printing the neural network architecture as a checkpoint

# Setting the range for the hyperparameter search
params = {
    # 'lr': [0.001, 0.01, 0.1],
    'module__kernel_size_conv1': list(range(2, 20)),
    'module__stride_conv1': list(range(1, 20)),
    'module__num_units': [1, 2, 3, 4],
    'optimizer__weight_decay': [0.001, 0.01, 0.1, 0],
}

gs = RandomizedSearchCV(net, params, n_iter=1000, refit=False, cv=5,
                        scoring='accuracy', verbose=1, n_jobs=-1)

gs.fit(X, y)
print(gs.best_score_, gs.best_params_)
