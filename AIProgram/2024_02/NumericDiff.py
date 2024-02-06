
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

# Models

# Image Processing / Computer Vision

# Optimization

# Auxiliary
from numba import jit, njit

# Visualization
import distinctipy
import matplotlib.colors

# Miscellaneous
from enum import auto, Enum, unique

# Typing
from typing import Callable, Dict, List, Tuple

# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()


def CalcFunGrad( vX: np.ndarray, hF: Callable, /, *, diffMode: DiffMode = DiffMode.CENTRAL, ε: float = 1e-6 ):
    """
    Calculates the gradient of `hF` using finite differences method (Numerical differentiation).
    Args:
        vX          - Vector (numElements, ) of the point to calculate the gradient at.
        hF          - A function to calculate the gradient for.
        diffMode    - The method to use for the numeric calculation.
        ε           - A positive float number as the "step size".
    Output:
        vG          - Vector of the gradient.
    """

    numElements = np.size(vX)

    objFunRef   = hF(vX)
    vG          = np.zeros_like(vX)
    vE          = np.zeros_like(vX)

    if (diffMode is DiffMode.BACKWARD):
        hGradFun = lambda vP: (objFunRef - hF(vX - vP)) / ε
    elif (diffMode is DiffMode.CENTRAL):
        hGradFun = lambda vP: (hF(vX + vP) - hF(vX - vP)) / (2 * ε)
    elif (diffMode is DiffMode.FORWARD):
        hGradFun = lambda vP: (hF(vX + vP) - objFunRef) / ε
    elif (diffMode is DiffMode.COMPLEX):
        hGradFun = lambda vP: np.imag(hF(vX + 1j * vP)) / ε
    else:
        raise ValueError('Invalid value for `diffMode` parameter')
    
    for ii in range(numElements):
        vE.flat[ii] = ε
        vG.flat[ii] = hGradFun(vE)
        vE.flat[ii] = 0.0
    
    return vG