
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

from numba import njit

# Machine Learning

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization

# Miscellaneous
from enum import auto, Enum, unique

# Typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Course Packages


# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

@unique
class ConvMode(Enum):
    # Type of data in the CSV
    FULL    = auto()
    SAME    = auto()
    VALID   = auto()

# Constants


# Optimization

# Model

# Type hints for SP Sparse: https://stackoverflow.com/questions/71501140
def GenConvMtx1D( vK: np.ndarray, numElements: int, /, *, convMode: ConvMode = ConvMode.FULL ) -> sp.sparse.csr.csr_matrix:

    """
    Generates a Convolution Matrix for 1D Kernel (The Vector vK) with support
    for different convolution shapes (Full / Same / Valid). The matrix is
    build such that for a signal 'vS' with 'numElements = size(vS, 1)' the 
    following are equivalent: 'mK * vS' and conv(vS, vK, convShapeString);
    """

    if (len(vK) <= numElements):
        kernelLength = len(vK)
        jjMax = numElements
        iiMax = kernelLength
        numCols = numElements
    else:
        kernelLength = numElements
        numElements = len(vK)
        jjMax = kernelLength
        iiMax = numElements
        numCols = kernelLength
    
    if convMode == ConvMode.FULL:
        rowIdxFirst = 1
        rowIdxLast  = numElements + kernelLength - 1
        outputSize  = numElements + kernelLength - 1
    elif convMode == ConvMode.SAME:
        rowIdxFirst = 1 + np.floor(kernelLength / 2)
        rowIdxLast  = rowIdxFirst + numElements - 1
        outputSize  = numElements
    elif convMode == ConvMode.VALID:
        rowIdxFirst = kernelLength
        rowIdxLast  = (numElements + kernelLength - 1) - kernelLength + 1
        outputSize  = numElements - kernelLength + 1
    

    mtxIdx = 0
    
    # The sparse matrix constructor ignores values of zero yet the Row / Column
    # indices must be valid indices (Positive integers). Hence 'vI' and 'vJ'
    # are initialized to 1 yet for invalid indices 'vV' will be 0 hence it has
    # no effect.
    
    vI = np.ones(shape = numElements * kernelLength)
    vJ = np.ones(shape = numElements * kernelLength)
    vV = np.zeros(shape = numElements * kernelLength)
    
    for jj in range(jjMax):
        for ii in range(iiMax):
            if ((ii + jj - 1 >= rowIdxFirst) and (ii + jj - 1 <= rowIdxLast)):
                # Valid output matrix row index
                mtxIdx = mtxIdx + 1
                vI[mtxIdx] = ii + jj - rowIdxFirst
                vJ[mtxIdx] = jj
                vV[mtxIdx] = vK(ii)
    
    
    mK = sparse(vI, vJ, vV, outputSize, numCols)
    
    return mK


