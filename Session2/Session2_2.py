from __future__ import division,print_function
import os
import json
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

from sklearn.metrics import confusion_matrix

import importlib
import bcolz
def save_array(fname, arr): 
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname): 
    return bcolz.open(fname)[:]

def onehot(x):
    """
    1d array to hot encoding.

    Example
    [1, 0, 0, 1] is mapped to [[0, 1], [1, 0], [1, 0], [0, 1]]
    """
    x_reshaped = x.reshape((-1,1))
    onehotenc = OneHotEncoder().fit_transform(x_reshaped)
    denseOneHot = onehotenc.todense()
    return np.array(denseOneHot)

#path = "data/dogscats/sample/"
path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size = 40 #max 64
if(not os.path.exists(model_path + 'valid_data.bc')):
   val_data = get_data(path + 'valid')
   save_array(model_path + 'valid_data.bc', val_data)

if(not os.path.exists(model_path + 'train_data.bc')):
   trn_data = get_data(path + 'train')
   save_array(model_path + 'train_data.bc', trn_data)

trn_data = load_array(model_path + 'train_data.bc')
val_data = load_array(model_path + 'valid_data.bc')

print(trn_data.shape)
print(val_data.shape)

# Use batch size of 1 since we're just doing preprocessing on the CPU
val_batches_for_classes = get_batches(path + 'valid', shuffle=False, batch_size=1)
batches_for_classes = get_batches(path+'train', shuffle=False, batch_size=1)

val_classes = val_batches_for_classes.classes #[0,0,0,1,1,1]
trn_classes = batches_for_classes.classes
val_labels = onehot(val_classes) #[[1,0][1,0][1,0][0,1][0,1][0,1]]
trn_labels = onehot(trn_classes)



from vgg16 import Vgg16
vgg = Vgg16()

vgg.model.summary()

vgg.model.pop()
for layer in vgg.model.layers: 
    layer.trainable = False

vgg.model.add(Dense(2, activation='softmax'))

gen = image.ImageDataGenerator()
batches = gen.flow(trn_data, trn_labels, batch_size=batch_size, shuffle=True)
val_batches = gen.flow(val_data, val_labels, batch_size=batch_size, shuffle=False)


opt = RMSprop(lr = 0.1)
vgg.model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])

vgg.model.fit_generator(batches, 
                                 steps_per_epoch=int(batches.n/batches.batch_size),
                                 epochs=2,
                                 validation_data=val_batches, 
                                 validation_steps=int(val_batches.n/val_batches.batch_size))

vgg.model.save_weights(model_path + 'finetune1.h5')
vgg.model.load_weights(model_path + 'finetune1.h5')
vgg.model.evaluate(val_data, val_labels)


preds = vgg.model.predict_classes(val_data, batch_size=batch_size)
probs = vgg.model.predict_proba(val_data, batch_size=batch_size)[:,0]
probs[:8]

cm = confusion_matrix(val_classes, preds)
plot_confusion_matrix(cm, {'cat':0, 'dog':1})
plt.show()


layers = vgg.model.layers
# Get the index of the first dense layer...
first_dense_idx = [index for index,layer in enumerate(layers) if type(layer) is Dense][0]
# ...and set this and all subsequent layers to trainable
for layer in layers[first_dense_idx:]: layer.trainable=True
K.set_value(opt.lr, 0.01)

vgg.model.fit_generator(batches, 
                                 steps_per_epoch=int(batches.n/batches.batch_size),
                                 epochs=3,
                                 validation_data=val_batches, 
                                 validation_steps=int(val_batches.n/val_batches.batch_size))

vgg.model.save_weights(model_path+'finetune2.h5')

vgg.model.evaluate(val_data, val_labels)

preds = vgg.model.predict_classes(val_data, batch_size=batch_size)


accuracy = len(np.where(preds == val_classes)[0]) / len(preds)

cm = confusion_matrix(val_classes, preds)
plot_confusion_matrix(cm, {'cat':0, 'dog':1})
plt.show()

####################

for layer in layers[12:]: layer.trainable=True
K.set_value(opt.lr, 0.001)

vgg.model.fit_generator(batches, 
                                 steps_per_epoch=int(batches.n/batches.batch_size),
                                 epochs=4,
                                 validation_data=val_batches, 
                                 validation_steps=int(val_batches.n/val_batches.batch_size))

vgg.model.save_weights(model_path+'finetune3.h5')
vgg.model.load_weights(model_path+'finetune2.h5')

vgg.model.evaluate_generator(get_batches('valid', gen, False, batch_size*2), val_batches.N)
