#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 18:04:43 2019

@author: raghav
"""

# train the Deep NN on the minerals dataset

import numpy as np
import matplotlib
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch import tanh
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, BatchNorm1d, Conv1d, MaxPool1d
from torch.nn.functional import relu, elu, relu6, sigmoid, softmax, leaky_relu
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# from memory_profiler import profile
from data_augmentator import *

matplotlib.use('Agg')
print("step completed")  # Checkpointing

# Loading the train test and validation data and reshaping for the Neural Net
# Reshaping the DATA for training and handling the dtypes
nchannels = 1
rows = X_train_final.shape[1]
print(X_train_final.shape)
x_train = X_train_final[:21770].astype('float32')
x_train = x_train.reshape((-1, nchannels, rows))
targets_train = y_train_final[:21770].astype('int32')

x_valid = X_test_validation[:653].astype('float32')
x_valid = x_valid.reshape((-1, nchannels, rows))
targets_valid = y_test_validation[:653].astype('int32')

# x_test = X_test_validation[:653].astype('float32')
# x_test = x_test.reshape((-1, nchannels, rows))
# targets_test = y_test_validation[:653].astype('int32')

print("Information on dataset")
print("x_train", x_train.shape)
print("targets_train", targets_train.shape)
print("x_valid", x_valid.shape)
print("targets_valid", targets_valid.shape)
# print("x_test", x_test.shape)
# print("targets_test", targets_test.shape)
print("x_train shape[1]", x_train.shape[1])
print("x_train shape[2]", x_train.shape[2])
# print("targets_test type:",targets_test.dtype)

print("DATA RESHAPED")
print("--------------")


# DEFINING THE NETWORK HERE
# hyperameters of the model
num_classes = len(set(y_train_final))
channels = x_train.shape[1]
height = x_train.shape[2]

num_filters_conv1 = 16
kernel_size_conv1 = 21 # [height, width]
stride_conv1 = 1 # [stride_height, stride_width]
padding_conv1 = 0

num_filters_conv2 = 32
kernel_size_conv2 = 11
stride_conv2 = 1
padding_conv2 = 0

num_filters_conv3 = 64
kernel_size_conv3 = 5
stride_conv3=1
padding_conv3=0

kernel_size_pooling1 = 2
stride_pooling1 = 2
padding_pooling1 = 0

num_l1 = 2048


def compute_conv_dim(dim_size, k, p, s):
    """
    computes the shape of the layers after convolution
    """
    return int((dim_size - k + 2 * p) / s + 1)


def compute_pool_dim(dim_size, kp, pp, sp):
    """
    computes the size of the layer after pooling
    """
    return int((dim_size - kp + 2 * pp) / sp + 1)


# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = Conv1d(in_channels=channels,
                             out_channels=num_filters_conv1,
                             kernel_size=kernel_size_conv1,
                             padding=padding_conv1,
                             stride=stride_conv1)
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.22)
        self.conv_1_bn = BatchNorm1d(num_filters_conv1)
        self.maxpool_1 = MaxPool1d(kernel_size=kernel_size_pooling1,
                                   stride=stride_pooling1)

        self.conv_height1 = compute_pool_dim(compute_conv_dim(height,kernel_size_conv1,
                                                                  padding_conv1,stride_conv1)
                                                 ,kernel_size_pooling1,
                                                padding_pooling1,stride_pooling1)
        self.conv_2 = Conv1d(in_channels=num_filters_conv1,
                             out_channels=num_filters_conv2,
                             kernel_size=kernel_size_conv2,
                             padding=padding_conv2,
                             stride=stride_conv2)
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.22)
        self.conv_2_bn = BatchNorm1d(num_filters_conv2)
        self.maxpool_2 = MaxPool1d(kernel_size=kernel_size_pooling1,
                                   stride=stride_pooling1)
        self.conv_height2 = compute_pool_dim(compute_conv_dim(self.conv_height1, kernel_size_conv2, padding_conv2, 
                                                                 stride_conv2),kernel_size_pooling1,
                                                padding_pooling1,stride_pooling1)

        self.conv_3 = Conv1d(in_channels=num_filters_conv2,
                             out_channels=num_filters_conv3,
                             kernel_size=kernel_size_conv3,
                             padding=padding_conv3,
                             stride=stride_conv3)
        torch.nn.init.normal_(self.conv_3.weight, mean=0, std=0.22)
        self.conv_3_bn = BatchNorm1d(num_filters_conv3)
        self.maxpool_3 = MaxPool1d(kernel_size=kernel_size_pooling1,
                                   stride=stride_pooling1)

        self.conv_out_height = compute_pool_dim(compute_conv_dim(self.conv_height2, kernel_size_conv3, padding_conv3, 
                                                                 stride_conv3),kernel_size_pooling1,
                                                padding_pooling1,stride_pooling1)
        # in features dimensions for the fully connected layer
        self.l1_in_features = num_filters_conv3 * self.conv_out_height

        self.l_1 = Linear(in_features=self.l1_in_features,
                          out_features=num_l1,
                          bias=True)
        torch.nn.init.normal_(self.l_1.weight, mean=0, std=0.22)
        self.l_1_bn = BatchNorm1d(num_l1)
        self.l_out = Linear(in_features=num_l1, 
                            out_features=num_classes,
                            bias=True)
        self.l_2_bn = BatchNorm1d(num_classes)
        self.dropout1d = nn.Dropout(0.5)

    def forward(self, x):  # x.size() = [batch, channel, height, width]
        x = leaky_relu(self.conv_1_bn(self.conv_1(x)))
        x = self.maxpool_1(x)

        x = leaky_relu(self.conv_2_bn(self.conv_2(x)))
        x = self.maxpool_2(x)

        x = leaky_relu(self.conv_3_bn(self.conv_3(x)))
        x = self.maxpool_3(x)

        # Returns a new tensor with the same data as the self tensor,
        # but of a different size.
        # the size -1 is inferred from other dimensions
        x = x.view(-1, self.l1_in_features)

        x = tanh(self.l_1_bn(self.l_1(x)))
        x = self.dropout1d(x)  # implemented dropout
        return softmax(self.l_2_bn(self.l_out(x)), dim=1)


net = Net()
print(net)

# Define the optimizer and the loss function
# Assigning weights to each class
class_weights = np.unique(targets_train, return_counts=True)[1]
# print(class_weights)
crossentropy_weights = torch.from_numpy(np.reciprocal(class_weights.astype('float32')))
# print(crossentropy_weights)

criterion = nn.CrossEntropyLoss(weight=crossentropy_weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Testing Forward Pass
x = np.random.normal(0, 1, (5, 1, 701)).astype('float32')
# print(x.shape)
out = net(Variable(torch.from_numpy(x)))
print(out.size(), out)
print("The forward pass is working just fine")

print("---------------TRAINING STARTS---------------")

batch_size = 1000
num_epochs = 50
num_samples_train = x_train.shape[0]
num_batches_train = int(np.ceil(num_samples_train / float(batch_size)))
num_samples_valid = x_valid.shape[0]
num_batches_valid = int(np.ceil(num_samples_valid / float(batch_size)))

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
losses = []
losses_valid = []
losses_train = []

get_slice = lambda i, size, num_samples: range(i * size, np.minimum((i + 1) * size,num_samples))

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    # Train
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size, num_samples_train)
        # print(slce)
        x_batch = Variable(torch.from_numpy(x_train[slce]))
        output = net(x_batch)

        # compute gradients given loss
        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss.data.numpy()
    losses.append(cur_loss / (batch_size*num_batches_train))

    net.eval()
    # Evaluate training
    train_preds, train_targs = [], []
    cur_loss = 0

    for i in range(num_batches_train):
        slce = get_slice(i, batch_size, num_samples_train)
        x_batch = Variable(torch.from_numpy(x_train[slce]))

        output = net(x_batch)
        preds = torch.max(output, 1)[1]

        train_targs += list(targets_train[slce])
        train_preds += list(preds.data.numpy())

        target_batch = Variable(torch.from_numpy(targets_train[slce]).long())
        batch_loss = criterion(output, target_batch)

        cur_loss += batch_loss.data.numpy()
    losses_train.append(cur_loss / (batch_size*num_batches_train))

    # Evaluate validation
    val_preds, val_targs = [], []
    cur_loss = 0
    for i in range(num_batches_valid):
        slce = get_slice(i, batch_size, num_samples_valid)
        x_batch = Variable(torch.from_numpy(x_valid[slce]))

        output = net(x_batch)
        preds = torch.max(output, 1)[1]

        val_preds += list(preds.data.numpy())
        val_targs += list(targets_valid[slce])

        target_batch = Variable(torch.from_numpy(targets_valid[slce]).long())
        batch_loss = criterion(output, target_batch)
        cur_loss += batch_loss.data.numpy()
    losses_valid.append(cur_loss / (batch_size*num_batches_valid))

    train_acc_cur = accuracy_score(train_targs, train_preds)
    valid_acc_cur = accuracy_score(val_targs, val_preds)

    train_acc.append(train_acc_cur)
    valid_acc.append(valid_acc_cur)

    if epoch % 1 == 0:  # if useful when plotting every 5th epoch or so
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.title('The Performance of the network while training')
plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
plt.legend(['Train Acc', 'Val Acc'])
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.savefig('ImportantPlots/NetworkPerformance.png')
plt.show()

plt.figure()
plt.title('The Average Loss function Plotted')
plt.plot(epoch, losses_train, 'r', label='Training Loss')
plt.plot(epoch, losses_valid, 'b', label='Validation Loss')
plt.legend(loc='upper right')
plt.savefig('ImportantPlots/AverageLossFunction.png')
plt.show()

val_preds_array = np.asarray(val_preds)
val_targs_array = np.asarray(val_targs)
train_preds_array = np.asarray(train_preds)
train_targs_array = np.asarray(train_targs)
train_accuracy_array = np.asarray(train_acc)
val_accuracy_array = np.asarray(valid_acc)
valid_loss_array = np.asarray(losses_valid)
train_loss_array = np.asarray(losses_train)
loss_array = np.asarray(losses)

np.save('ImportantVariables/Loss', loss_array)
np.save('ImportantVariables/Validation_Loss', valid_loss_array)
np.save('ImportantVariables/Training_Loss', train_loss_array)
np.save('ImportantVariables/Validation_Predictions', val_preds_array)
np.save('ImportantVariables/Validation_targs', val_targs_array)
np.save('ImportantVariables/Training_Predictions', train_preds_array)
np.save('ImportantVariables/Training_targs', train_targs_array)
np.save('ImportantVariables/Validation_Accuracy', val_accuracy_array)
np.save('ImportantVariables/Training_Accuracy', train_accuracy_array)

torch.save(net.state_dict(), 'DEEP_CNN_TRAINEDMODEL/Final_Model.pth')
