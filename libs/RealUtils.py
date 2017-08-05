from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt

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

def plots_SingleChannel(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 1):
            ims = ims.transpose((0,2,3,1)) #=> (-1,28,28,1)
    x1 = ims.shape[1]
    x2 = ims.shape[2]
    ims = ims.reshape((-1, x1, x2))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims) // rows, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

    plt.show()
