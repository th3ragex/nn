from RealUtils import *
import numpy as np
from numpy import genfromtxt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

import pandas as pd

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

def norm_input(x): 
    return (x - mean_px) / std_px

def GetSteps(batch):
    return batch.n / batch.batch_size

K.set_image_data_format('channels_first')
batch_size = 250
max_rows = None
train_proportion = 0.80

#Read files
#rawdataTrain = genfromtxt(os.getcwd()+'\\Session3\\MNISTKaggle\\train.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)
#rawdataSubmission = genfromtxt(os.getcwd()+'\\Session3\\MNISTKaggle\\test.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)

rawdataTrain = genfromtxt('E:\\train.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)
rawdataSubmission = genfromtxt('E:\\test.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_rows)

#Fix dimensions => x,1,28,28
train_data = rawdataTrain[:, 1: ].reshape((-1, 1, 28, 28))
train_labels = rawdataTrain[:, 0:1].reshape((-1))
x_submission = rawdataSubmission.reshape((-1, 1, 28, 28))

#Split in test and train
train_inds,test_inds = get_train_test_inds(train_labels, train_proportion=train_proportion)

Y_train = train_labels[train_inds]
Y_test = train_labels[test_inds]

X_train = train_data[train_inds]
X_test = train_data[test_inds]

#plt.imshow(X_train[3][0])
#plt.show()
#plt.imshow(Y_test[3])
#plt.show()


#onehot
Y_train = onehot(Y_train)
Y_test = onehot(Y_test)

#plt.hist(Y_train)
#plt.show()
#plt.hist(Y_test)
#plt.show()
#print(X_train.shape)
#print(Y_train.shape)
#print(x_submission.shape)


def get_model():
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

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

'''
                            (rotation_range=8., 
                               width_shift_range=0.08, 
                               shear_range=0.3, 
                               height_shift_range=0.08, 
                               zoom_range=0.08,
'''

imageGen = image.ImageDataGenerator(data_format='channels_first')

train_batches = imageGen.flow(X_train, Y_train, batch_size=batch_size)
test_batches = imageGen.flow(X_test,  Y_test,  batch_size=batch_size, shuffle=False)


model = get_model()

model.optimizer.lr = 0.1
model.fit_generator(train_batches, GetSteps(train_batches), epochs=4, validation_data = test_batches, validation_steps = GetSteps(test_batches))

model.optimizer.lr = 0.01
model.fit_generator(train_batches, GetSteps(train_batches), epochs=10, validation_data = test_batches, validation_steps = GetSteps(test_batches))

model.optimizer.lr = 0.001
model.fit_generator(train_batches, GetSteps(train_batches), epochs=18, validation_data = test_batches, validation_steps = GetSteps(test_batches))


submissionLabels = model.predict_classes(x_submission, batch_size)

#plots_SingleChannel(x_submission[:12], rows=2)

id = np.array(range(0, len(submissionLabels))) + 1
label = submissionLabels

df = pd.DataFrame({'ImageId':id, 'Label':label})
from time import gmtime, strftime
curTime = strftime("%Y_%m_%d %H_%M_%S", gmtime())
df.to_csv(os.getcwd()+'\\Session3\\MNISTKaggle\\submission'+curTime+'.csv', index=False)

#https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model

model.save(os.getcwd()+'\\Session3\\MNISTKaggle\\model'+curTime+'2.h5')
model.save_weights(model.save(os.getcwd()+'\\Session3\\MNISTKaggle\\model weights'+curTime+'.h5'))