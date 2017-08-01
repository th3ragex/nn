from __future__ import division,print_function
import os, json
from glob import glob

import numpy as np
from numpy.random import random, permutation
np.set_printoptions(precision=4, linewidth=100)

import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import utils
from utils import plots, get_batches, plot_confusion_matrix, get_data

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

x = random((30,2))
y = np.dot(x, [2., 3.]) + 1. # [2, 3] = weights for feature 1 and feature 2

lm = Sequential([ Dense(1, input_shape=(2,)) ])
lm.compile(optimizer=SGD(lr=0.1), loss='mse')


print(lm.evaluate(x, y, verbose=0))

lm.fit(x, y, epochs=5, batch_size=1)

print(lm.evaluate(x, y, verbose=0))

print(lm.get_weights())