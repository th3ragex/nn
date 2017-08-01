from RealUtils import *
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.datasets import mnist

K.set_image_data_format('channels_first')
batch_size=64

(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Append dimension for color channel  (60000,28,28) => (60000,1,28,28)
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


#Linear model

gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)
test_batches = gen.flow(X_test, y_test, batch_size=64)

def get_lin_model():
    model = Sequential(
        [
            Lambda(norm_input, input_shape=(1, 28, 28), output_shape=(1, 28, 28)),
            Flatten(),
            Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

lm = get_lin_model()

#fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, 
##                    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, initial_epoch=0)
lm.optimizer.lr=0.001 #default
lm.fit_generator(batches, GetSteps(batches), epochs=1, validation_data = test_batches, validation_steps = GetSteps(test_batches))
#938/937 [==============================] - 6s - loss: 0.4297 - acc: 0.8740 - val_loss: 0.3003 - val_acc: 0.9151

lm.optimizer.lr=0.1
lm.fit_generator(batches, GetSteps(batches), epochs=1, validation_data = test_batches, validation_steps = GetSteps(test_batches))
#938/937 [==============================] - 6s - loss: 0.2996 - acc: 0.9154 - val_loss: 0.2802 - val_acc: 0.9204

lm.optimizer.lr=0.01
lm.fit_generator(batches, GetSteps(batches), epochs=4, validation_data = test_batches, validation_steps = GetSteps(test_batches))
#938/937 [==============================] - 6s - loss: 0.2684 - acc: 0.9248 - val_loss: 0.2851 - val_acc: 0.9196