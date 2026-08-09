import numpy as np 

def relu(x):
    x = np.asarray(x, dtype=float)
    relu_compute = np.maximum(0, x)
    return relu_compute