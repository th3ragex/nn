from numpy import genfromtxt
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import  model_selection

def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds


max_rows = 500

rawdataTrain = genfromtxt(os.getcwd()+'\\Session3\\MNISTKaggle\\train.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)
rawdataTest = genfromtxt(os.getcwd()+'\\Session3\\MNISTKaggle\\test.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)

x_train = rawdataTrain[:, 1: ].reshape((-1, 1, 28, 28))
y = rawdataTrain[:, 0:1].reshape((-1))

x_submission = rawdataTest


train_inds,test_inds = get_train_test_inds(y, train_proportion=0.80)

Y_train = y[train_inds]
Y_test = y[test_inds]

plt.hist(Y_train)
plt.show()
plt.hist(Y_test)
plt.show()

print(x_train.shape)
print(y_train.shape)
print(x_submission.shape)