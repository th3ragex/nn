
#from __future__ import division, print_function
#import utils;# reload(utils)
#from utils import *
from RealUtils import *
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

K.set_image_data_format('channels_first')
batch_size = 64

(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Append dimension for color channel (60000,28,28) => (60000,1,28,28)
X_test = np.expand_dims(X_test,1)
X_train = np.expand_dims(X_train,1)

X_train.shape

# Peek labels
y_train[:5]

#onehot
y_train = onehot(y_train)
y_test = onehot(y_test)

y_train[:5]

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def norm_input(x): 
    return (x - mean_px) / std_px

def GetSteps(batch):
    return batch.n / batch.batch_size

#Data augmentation

gen = image.ImageDataGenerator(rotation_range=8., width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08, data_format='channels_first')
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)

#Batchnorm + dropout + data augmentation

def get_model_bn_do():
    model = Sequential(
        [
            Lambda(norm_input, input_shape=(1, 28, 28), output_shape=(1, 28, 28)),
            Convolution2D(32,(3,3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(32,(3,3), activation='relu'),
            MaxPooling2D(),
            BatchNormalization(axis=1),
            Convolution2D(64,(3,3), activation='relu'),
            BatchNormalization(axis=1),
            Convolution2D(64,(3,3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = get_model_bn_do()

model.fit_generator(batches, GetSteps(batches), epochs=1, validation_data = test_batches, validation_steps = GetSteps(test_batches))

model.optimizer.lr = 0.1
model.fit_generator(batches, GetSteps(batches), epochs=4, validation_data = test_batches, validation_steps = GetSteps(test_batches))

model.optimizer.lr = 0.01
model.fit_generator(batches, GetSteps(batches), epochs=12, validation_data = test_batches, validation_steps = GetSteps(test_batches))

model.optimizer.lr = 0.001
model.fit_generator(batches, GetSteps(batches), epochs=5, validation_data = test_batches, validation_steps = GetSteps(test_batches))
