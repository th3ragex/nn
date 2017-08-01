from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt
import utils

from utils import plots


def pred_batch2(model, imgs):
    preds = model.predict(imgs, False)
    idxs = np.argmax(preds, axis=1)

  
    print('Shape: {}'.format(preds[0].shape))
    print('Shape: {}'.format(preds[1].shape))
    print('Probabilities: {}'.format(preds[0]))
    print('Classes: {}\n'.format(preds[1]))
    print('Predictions prob/class: ')
    
   

    for i in range(len(preds[0])):
         print ('  {:.4f}/{}'.format(preds[0][i], model.classes[preds[1][i]]))
    return preds

path = "data/dogscats/"
#path = "data/dogscats/sample/"

# As large as you can, but no larger than 64 is recommended. 
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size=64

# Import our class, and instantiate

# Upgrade
#import vgg16; reload(vgg16)
import vgg16
from vgg16 import Vgg16


vgg = Vgg16()
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'train', batch_size=batch_size)
val_batches = vgg.get_batches(path+'valid', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)


# Predict 4 images
test = vgg.get_batches(path+'valid', batch_size=4)
imgs,labels = next(test)

# This shows the 'ground truth'
plots(imgs, titles=labels)
plt.show()


pred = pred_batch2(vgg, imgs)

