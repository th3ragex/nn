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

#path = "data/dogscats/sample/"
path = "data/dogscats/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)

batch_size=64

# Ensure that we return to theano dimension ordering
#K.set_image_dim_ordering('th')

#ol_format = K.image_data_format



# dim_ordering='tf' uses tensorflow dimension ordering,
#   which is the same order as matplotlib uses for display.
# Therefore when just using for display purposes, this is more convenient
gen = image.ImageDataGenerator(featurewise_center=False,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         zca_whitening=True,
                         rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.15,
                         zoom_range=0.1,
                         channel_shift_range=10.,
                         fill_mode='nearest',
                         cval=0.,
                         horizontal_flip=True,
                         vertical_flip=False,
                         rescale=None,
                         preprocessing_function=None,
                         data_format='channels_last')

#rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
#width_zoom_range=0.2, shear_range=0.15, zoom_range=0.1, 
       #channel_shift_range=10., horizontal_flip=True, dim_ordering='tf'

# Create a 'batch' of a single image
i = ndimage.imread('data/dogscats/test/7.jpg')
i2 = ndimage.imread('data/dogscats/test/7_2.jpg')
img = np.expand_dims(i,0)
img2 = np.expand_dims(i2,0)


# (2, width, height, channels)
imgs = np.concatenate((img, img2))

# Request the generator to create batches from this image
# This returns (n, with, height, channels)
# n is the number of images available
# The returned images are randomly augmented and shuffled.
aug_iter = gen.flow(imgs)

# Get eight examples of these augmented images
aug_imgs = [next(aug_iter) for i in range(8)]

