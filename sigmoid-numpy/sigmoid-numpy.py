import numpy as np 


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    neg_x = -x
    exp_neg_x = np.exp(neg_x)
    denominator = 1 + exp_neg_x
    result = 1 / denominator
    return result