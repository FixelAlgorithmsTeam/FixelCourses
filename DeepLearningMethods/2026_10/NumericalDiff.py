
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
from numpy.typing import NDArray

# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()


def CalcFunGrad( vX: NDArray, hF: Callable, /, *, diffMode: DiffMode = DiffMode.CENTRAL, ε: float = 1e-6 ) -> NDArray:
    """
    Calculates the gradient of `hF` using finite differences method (Numerical differentiation).
    Input:
        vX          - Vector (numElements, ) of the point to calculate the gradient at.
        hF          - A function to calculate the gradient for.
        diffMode    - The method to use for the numeric calculation.
        ε           - A positive float number as the "step size".
    Output:
        vG          - Vector of the gradient at `hF(vX)`.
    """

    numElements = np.size(vX)

    objFunRef   = hF(vX)
    vG          = np.zeros_like(vX)
    vE          = np.zeros_like(vX)

    # Auxiliary function to calculate the directional derivative by
    if (diffMode is DiffMode.BACKWARD):
        hDirDerv = lambda vEi: (objFunRef - hF(vX - vEi)) / ε
    elif (diffMode is DiffMode.CENTRAL):
        hDirDerv = lambda vEi: (hF(vX + vEi) - hF(vX - vEi)) / (2 * ε)
    elif (diffMode is DiffMode.FORWARD):
        hDirDerv = lambda vEi: (hF(vX + vEi) - objFunRef) / ε
    elif (diffMode is DiffMode.COMPLEX):
        hDirDerv = lambda vEi: np.imag(hF(vX + 1j * vEi)) / ε
    else:
        raise ValueError('Invalid value for `diffMode` parameter')
    
    # Iterate over all directions
    for ii in range(numElements):
        vE.flat[ii] = ε            #<! Generate the basis direction (`vEi`)
        vG.flat[ii] = hDirDerv(vE) #<! Calculate the directional derivative
        vE.flat[ii] = 0.0
    
    return vG