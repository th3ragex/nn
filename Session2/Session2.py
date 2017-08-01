from __future__ import division,print_function
import os, json
from glob import glob
import matplotlib.gridspec as gridspec

import numpy as np
from numpy.random import random, permutation
np.set_printoptions(precision=10, linewidth=100)

import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from sklearn.preprocessing import OneHotEncoder

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

#import importlib
#importlib.reload(ModelExploration)
import ModelExploration 



import bcolz
def save_array(fname, arr): 
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname): 
    return bcolz.open(fname)[:]


def onehot(x): 
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())

#path = "data/dogscats/sample/"
path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=100
#batch_size=4

# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches = get_batches(path+'valid', shuffle=False, batch_size=1)
batches = get_batches(path+'train', shuffle=False, batch_size=1)

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model

if(not os.path.exists(model_path + 'valid_data.bc')):
   val_data = get_data(path+'valid')
   save_array(model_path + 'valid_data.bc', val_data)

if(not os.path.exists(model_path + 'train_data.bc')):
   trn_data = get_data(path+'train')
   save_array(model_path+ 'train_data.bc', trn_data)

trn_data = load_array(model_path+'train_data.bc')
val_data = load_array(model_path+'valid_data.bc')

print(trn_data.shape)
print(val_data.shape)

val_classes = val_batches.classes #[0,0,0,1,1,1]
trn_classes = batches.classes
val_labels = onehot(val_classes) #[[1,0][1,0][1,0][0,1][0,1][0,1]]
trn_labels = onehot(trn_classes)

lines = trn_data.shape[0]


if(not os.path.exists(model_path+ 'train_lastlayer_features.bc')):
    trn_features = model.predict(trn_data, batch_size=batch_size, verbose=1)
    save_array(model_path+ 'train_lastlayer_features.bc', trn_features)

if(not os.path.exists(model_path+ 'valid_lastlayer_features.bc')):
    val_features = model.predict(val_data, batch_size=batch_size, verbose=1)
    save_array(model_path + 'valid_lastlayer_features.bc', val_features)

val_features = load_array(model_path+'valid_lastlayer_features.bc')    
trn_features = load_array(model_path+'train_lastlayer_features.bc')

# 1000 inputs, since that's the saved features, and 2 outputs, for dog and cat
lm = Sequential([ Dense(2, activation='softmax', input_shape=(1000,)) ])
lm.compile(optimizer=RMSprop(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# The most important metrics for us to look at are for the validation set, since we want to check for over-fitting. 
batch_size=4
lm.fit(trn_features, trn_labels, epochs=3, batch_size=batch_size, validation_data=(val_features, val_labels))

print(lm.summary())


# We want both the classes...
predicted_labels = lm.predict_classes(val_features, batch_size=batch_size)
# ...and the probabilities of being a cat
predicted_probs = lm.predict_proba(val_features, batch_size=batch_size)

# Explore the model
# Labels: 0 = cat, 1 = dog
# predicted_label is also the column in the probabilities column
validation_file_paths = [path + 'valid/' + name  for name in val_batches.filenames]
actual_labels = val_labels.argmax(axis = -1) # one encoded -> label 

importlib.reload(ModelExploration)
import ModelExploration as me
# 8.27 × 11.7
f = plt.figure(figsize=(11.7,8.27))
me.Explore(batches.class_indices, actual_labels, predicted_labels, predicted_probs, validation_file_paths)

