# ==============================================
# from typing import Callable
# ==============================================
#
# 
# https://stackoverflow.com/questions/70967266/what-exactly-is-python-typing-callable
# 
# I would like to add to other answers that starting with Python 3.9, 
# typing.Callable is deprecated.
#  You should use collections.abc.Callable instead.
# 
# For more details and the rationale, 
# you can take a look at PEP 585
# 
# https://peps.python.org/pep-0585/
# ==============================================
# ==============================================


#+++++++++++++++++++++++++++++++++++++++++++++++
# importing modules and libraries
#+++++++++++++++++++++++++++++++++++++++++++++++
from collections.abc import Callable
from typing import List

import numpy as np
from numpy import ndarray




Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]
#+++++++++++++++++++++++++++++++++++++++++++++++

#===============================================
# 
# 
# 
# ==============================================
def deriv(
        func: Callable[[np.ndarray], np.ndarray],
        input_: np.ndarray,
        delta: float= 0.001 )-> np.ndarray:
    '''
    Evaluates the estimated derivative of a function "func" at every element in
    "input_" array
    '''
    return (
            func(input_ + delta)-
            func(input_ - delta)
            ) / (2*delta)

#===============================================
# 
# 
# 
# ==============================================
def nested_functions(chain: Chain,
                     x: ndarray)-> ndarray:
    assert len(chain)>0 , \
    "There must be a function to compute its output on the input value"
    out_ = x
    for i in range(len(chain)):
        tmp = chain[i]
        out_ = tmp(out_)
    
    return out_
#===============================================
# 
# 
# 
#===============================================
def chain_deriv(chain: Chain,
                input_range: ndarray):
    if len(chain) == 2:
        return deriv(chain[0], input_range) *\
               deriv(chain[1], 
                     chain[0](input_range))
    else:
        nested = nested_functions(chain[:-1], 
                                  input_range)

        return deriv(chain[-1], nested) * \
               chain_deriv(chain=chain[:-1], 
                           input_range=input_range)
#===============================================
# plotting a chain of functions
# 
# 
#===============================================
def plot_chain(ax,
               chain: Chain, 
               input_range: ndarray) -> None:
    '''
    Plots a chain function - a function made up of 
    multiple consecutive ndarray -> ndarray mappings - 
    Across the input_range
    
    ax: matplotlib Subplot for plotting
    '''
    
    assert input_range.ndim == 1, \
    "Function requires a 1 dimensional ndarray as input_range"

    output_range = nested_functions(chain, 
                                    input_range)
    ax.plot(input_range, output_range)
#===============================================
def plot_chain_deriv(ax,
                     chain: Chain,
                     input_range: ndarray) -> ndarray:
    '''
    Uses the chain rule to plot the derivative of a function consisting of two nested functions.
    
    ax: matplotlib Subplot for plotting
    '''
    output_range = chain_deriv(chain, input_range)
    ax.plot(input_range, output_range)

#-----------------------------------------------
# End of The utils_1.py
#-----------------------------------------------
