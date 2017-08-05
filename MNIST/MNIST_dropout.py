from RealUtils import *
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from MNIST_shared import *

K.set_image_data_format('channels_first')
batch_size = 64
max_samples = None
train_proportion = 0.80

X_train, Y_train, X_test, Y_test, X_submission = load_mnist_data(max_samples, train_proportion)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def norm_input(x): 
    return (x - mean_px) / std_px

def get_model():
    model = Sequential(
        [
            Lambda(norm_input, input_shape=(1, 28, 28), output_shape=(1, 28, 28)),
            Convolution2D(32,(3,3), activation='relu'),
            Convolution2D(32,(3,3), activation='relu'),
            MaxPooling2D(),
            Convolution2D(64,(3,3), activation='relu'),
            Convolution2D(64,(3,3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


imageGen = image.ImageDataGenerator(data_format='channels_first')
train_batches = imageGen.flow(X_train, Y_train, batch_size=batch_size)
test_batches = imageGen.flow(X_test,  Y_test,  batch_size=batch_size, shuffle=False)



model = get_model()
fit_it(model, train_batches, test_batches, epochs=2, lr=0.001)
fit_it(model, train_batches, test_batches, epochs=2, lr=0.0001)
#525/524 [==============================] - 15s - loss: 0.0325 - acc: 0.9894 - val_loss: 0.0353 - val_acc: 0.9901
#525/524 [==============================] - 15s - loss: 0.0237 - acc: 0.9925 - val_loss: 0.0287 - val_acc: 0.9920
#525/524 [==============================] - 15s - loss: 0.0163 - acc: 0.9949 - val_loss: 0.0353 - val_acc: 0.9907


# Summary


# Predict and submit
submissionLabels = model.predict_classes(X_submission, batch_size)
write_submission_file(submissionLabels)


#https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
model.save(os.getcwd() + '\\Session3\\MNISTKaggle\\model' + curTime + '2.h5')
model.save_weights(model.save(os.getcwd() + '\\Session3\\MNISTKaggle\\model weights' + curTime + '.h5'))