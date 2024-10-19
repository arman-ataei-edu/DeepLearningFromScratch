#+++++++++++++++++++++++++++++++++++++++++++++++
# importing modules and libraries
#+++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy import ndarray
#+++++++++++++++++++++++++++++++++++++++++++++++

#===============================================
# Activation Functions
# 
# 
# ==============================================
def square(x: ndarray) -> ndarray:
    '''
    Square each element in the input ndarray.
    '''
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray
    '''
    return np.maximum(0.2 * x, x)

def sigmoid(x: ndarray) -> ndarray:
    '''
    Apply the sigmoid function to each element in the input ndarray.
    '''
    return 1 / (1 + np.exp(-x))

def linear(x: ndarray) -> ndarray:
    '''
    Linear activation function, return the input unchanged.
    '''
    return x

def tanhip(x: ndarray) -> ndarray:
    '''
    Aplly the tangent hyperbolic to each element in the input ndarray.
    '''
    return np.tanh(x)



#-----------------------------------------------
# End of The activations.py
#-----------------------------------------------
