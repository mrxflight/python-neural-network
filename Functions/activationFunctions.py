import numpy as np

#activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_prime(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds