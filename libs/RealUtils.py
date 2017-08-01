from sklearn.preprocessing import OneHotEncoder
import numpy as np


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