from time import gmtime, strftime
import numpy as np
from numpy import genfromtxt
import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder

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


def get_train_test_idx(y,train_proportion=0.7):
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

def GetSteps(batch):
    return batch.n / batch.batch_size



def load_mnist_data(max_samples = None, train_proportion = 0.8):

    data_path = 'E:\\Data\\'

    print('Loading start at '+strftime("%H:%M:%S", gmtime()))
    
    rawdataTrain = genfromtxt(data_path + 'train.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_samples)
    rawdataSubmission = genfromtxt(data_path + 'test.csv', delimiter=',', skip_header=1, dtype = np.uint8, max_rows = max_samples)
    
    print('Finished start at '+strftime("%H:%M:%S", gmtime()))

    #Fix dimensions => x,1,28,28
    train_data = rawdataTrain[:, 1: ].reshape((-1, 1, 28, 28))
    train_labels = rawdataTrain[:, 0:1].reshape((-1))
    submission_data = rawdataSubmission.reshape((-1, 1, 28, 28))

    #Split in test and train
    train_idx, test_idx = get_train_test_idx(train_labels, train_proportion=train_proportion)

    X_test = train_data[test_idx]
    Y_test = train_labels[test_idx]
    X_train = train_data[train_idx]
    Y_train = train_labels[train_idx]

    #onehot
    Y_train = onehot(Y_train)
    Y_test = onehot(Y_test)

    return X_train, Y_train, X_test, Y_test, submission_data



def write_submission_file(labels):

    id = np.array(range(0, len(labels))) + 1
    df = pd.DataFrame({'ImageId':id, 'Label':labels})

   
    curTime = strftime("%Y_%m_%d %H_%M_%S", gmtime())
    df.to_csv(os.getcwd()+ '\\MNIST\\Data\\submission'+curTime+'.csv', index=False)
    


def fit_it(self, generator, validation_generator, epochs = 1, verbose = 1, callbacks = None,
    class_weight = None, max_q_size = 10, workers = 1, pickle_safe = False, initial_epoch = 0, lr = None):

    steps_per_epoch = generator.n / generator.batch_size        
    validation_steps = validation_generator.n / validation_generator.batch_size
    
    if not (lr == None):
        self.optimizer.lr = lr

    return self.fit_generator(generator, steps_per_epoch, epochs=epochs, verbose=verbose, callbacks=callbacks, 
                              validation_data=validation_generator, validation_steps=validation_steps, class_weight=class_weight, 
                              max_q_size=max_q_size, workers=workers, pickle_safe=pickle_safe,
                              initial_epoch=initial_epoch)


#Dump
'''
                            (rotation_range=8., 
                               width_shift_range=0.08, 
                               shear_range=0.3, 
                               height_shift_range=0.08, 
                               zoom_range=0.08,
'''
