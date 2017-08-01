import matplotlib.pyplot as plt
import math as m
import numpy as np

#x1, x2
values = np.array([10, 20, 30])
values_x = range(0, len(values))
#values = np.array(range(0,100))

def softmax(x, i):
    return m.exp(i) / np.sum(np.exp(x))

softmax_values = [softmax(values, i) for i in values]

#plt.plot(values, values)
print(softmax_values)
print(np.sum(softmax_values))
plt.plot(values_x, softmax_values)
plt.show()