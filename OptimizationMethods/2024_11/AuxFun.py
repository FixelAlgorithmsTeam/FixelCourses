
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
class ConvMode(Enum):
    # Convolution mode / shape
    FULL    = auto()
    SAME    = auto()
    VALID   = auto()


@unique
class DiffMode(Enum):
    # Numerical differentiation mode
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

@unique
class StepSizeMode(Enum):
    # Step size policy in Gradient Descent
    ADAPTIVE    = auto()
    CONSTANT    = auto()

# Constants


# Optimization

class GradientDescent():
    def __init__( self, vX: np.ndarray, hGradFun: Callable, μ: float, /, *, stepSizeMode: StepSizeMode = StepSizeMode.CONSTANT, hObjFub: Callable = None, α: float = 0.5 ) -> None:
        
        dataDim = len(vX)
        
        self._dataDim       = dataDim
        self._hGradFun      = hGradFun
        self.μ              = μ #<! Step Size
        self._stepSizeMode  = stepSizeMode #<! Step Size Mode
        self._hObjFub       = hObjFub #<! Objective function
        self.α              = α #<! Backtracking constant

        self.vX = np.copy(vX) #<! Current State
        self.vG = np.empty_like(vX) #<! Current Gradient
        self.vZ = np.empty_like(vX) #<! Buffer
        self.ii = 1
        
        pass

    # @njit
    def _ApplyIterationAdaptive( self ) -> np.ndarray:

        self.vG     = self._hGradFun(self.vX)
        currObjVal  = self._hObjFub(self.vX)
        self.vZ     = self.vX - self.μ * self.vG

        while(self._hObjFub(self.vZ) > currObjVal):
            # For production code, must be limited by value of `self.μ` and number iterations
            self.μ *= self.α
            self.vZ = self.vX - self.μ * self.vG
        
        self.vG *= self.μ
        self.μ   = max(1e-9, self.μ)
        self.μ  /= self.α

        self.vX -= self.vG #<! The gradient is pre scaled

        self.ii += 1

        return self.vX
    
    # @njit
    def ApplyIteration( self ) -> np.ndarray:

        if self._stepSizeMode == StepSizeMode.ADAPTIVE:
            return self._ApplyIterationAdaptive()
        
        self.vG  = self._hGradFun(self.vX)
        self.vX -= self.μ * self.vG

        self.ii += 1

        return self.vX
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List]:

        if logArg:
            lX = [None] * numIterations
            lX[0] = np.copy(self.vX)
        else:
            lX = None
        
        for jj in range(1, numIterations):
            vX = self.ApplyIteration()
            if logArg:
                lX[jj] = np.copy(vX)
        
        return lX

class ProxGradientDescent():
    def __init__( self, vX: np.ndarray, hGradFun: Callable, μ: float, λ: float, /, *, hProxFun: Callable = np.array, useAccel: bool = False ) -> None:
        
        dataDim = len(vX)
        
        self._dataDim = dataDim
        self._hGradFun = hGradFun
        self.μ = μ #<! Step Size
        self.λ = λ #<! Parameter Lambda
        self._hProxFun = hProxFun
        self.useAccel = useAccel

        self.vX     = np.copy(vX) #<! Current State
        self._vX_1  = np.copy(vX) #<! Previous state
        self.vG     = np.empty_like(vX) #<! Current Gradient
        self.vV     = np.empty_like(vX) #<! Buffer
        self.ii     = 1
        
        pass

    # @njit
    def _ApplyIterationFista( self ) -> np.ndarray:

        self.vV = self.vX + ((self.ii - 1) / (self.ii + 2)) * (self.vX - self._vX_1)
        self.vG = self._hGradFun(self.vV)
        
        self._vX_1[:] = self.vX[:]
        
        self.vX = self.vV - self.μ * self.vG
        self.vX = self._hProxFun(self.vX, self.μ * self.λ)

        self.ii += 1

        return self.vX
    
    # @njit
    def ApplyIteration( self ) -> np.ndarray:

        if self.useAccel:
            return self._ApplyIterationFista()
        
        self.vG  = self._hGradFun(self.vX)
        self.vX -= self.μ * self.vG
        self.vX  = self._hProxFun(self.vX, self.μ * self.λ)

        self.ii += 1

        return self.vX
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List]:

        if logArg:
            lX = [None] * numIterations
            lX[0] = np.copy(self.vX)
        else:
            lX = None
        
        for jj in range(1, numIterations):
            vX = self.ApplyIteration()
            if logArg:
                lX[jj] = np.copy(vX)
        
        return lX


# Model

# Type hints for SP Sparse: https://stackoverflow.com/questions/71501140
# @njit 
def GenConvMtx1D( vK: np.ndarray, numElements: int, /, *, convMode: ConvMode = ConvMode.FULL ) -> sp.sparse.csr.csr_matrix:

    """
    Generates a Convolution Matrix for 1D Kernel (The Vector vK) with support
    for different convolution shapes (Full / Same / Valid). The matrix is
    build such that for a signal (Vector) 'vS' with 'numElements = len(vS)' the 
    following are equivalent: 'mK @ vS' and `np.convolve(vS, vK, convModeStr)`.
    Input:
      - vK                -   Input 1D Convolution Kernel.
                              Structure: Vector.
                              Type: 'Single' / 'Double'.
                              Range: (-inf, inf).
      - numElements       -   Number of Elements.
                              Number of elements of the vector to be
                              convolved with the matrix. Basically set the
                              number of columns of the Convolution Matrix.
                              Structure: Scalar.
                              Type: 'int'.
                              Range: {1, 2, 3, ...}.
      - convShape         -   Convolution Shape.
                              The shape of the convolution which the output
                              convolution matrix should represent. The
                              options should match MATLAB's `conv2()` function
                              - Full / Same / Valid.
                              Structure: Scalar.
                              Type: 'Single' / 'Double'.
                              Range: {1, 2, 3}.
    Output:
      - mK                -   Convolution Matrix.
                              The output convolution matrix. The product of
                              'mK' and a vector 'vS' ('mK * vS') is the
                              convolution between 'vK' and 'vS' with the
                              corresponding convolution shape.
                              Structure: Matrix (Sparse).
                              Type: 'Single' / 'Double'.
                              Range: (-inf, inf).
    References:
      1.  Fixel's MATLAB function `CreateConvMtx1D()`.
    Remarks:
      1.  The output matrix is sparse data type in order to make the
          multiplication by vectors to more efficient.
      2.  In case the same convolution is applied on many vectors, stacking
          them into a matrix (Each signal as a vector) and applying
          convolution on each column by matrix multiplication might be more
          efficient than applying classic convolution per column.
      3.  The implementation matches MATLAB's `conv()`.  
          It differs from NumPy's `convolove()` which always use the shorter  
          input as the kernel which means it is commutative for any mode.  
          SciPy's `convolove()` matches MATLAB's `same` mode yet matches 
          NumPy's implementation in `valid` mode.
      4.  SciPy adds repetitive indices in: `mK[vI[k], vJ[k]] += vV[k]`.
          This is similar to MATLAB.  
      5.  SciPy does not remove explicit zeros. If `vV[k] == 0` it will
          be registered as an element in the matrix. Unlike MATLAB.
    TODO:
      1.  
      Release Notes:
      -   1.0.000     27/09/2024  Royi Avital
          *   First release version.
    """

    if (len(vK) <= numElements):
        # The case it matches NumPy / SciPy
        kernelLength = len(vK)
        jjMax        = numElements
        iiMax        = kernelLength
        numCols      = numElements
    else:
        kernelLength = numElements
        numElements  = len(vK)
        jjMax        = kernelLength
        iiMax        = numElements
        numCols      = kernelLength
    
    if convMode == ConvMode.FULL:
        rowIdxFirst = 0
        rowIdxLast  = numElements + kernelLength - 1
        outputSize  = numElements + kernelLength - 1
    elif convMode == ConvMode.SAME:
        rowIdxFirst = np.floor((kernelLength - 1)/ 2)
        rowIdxLast  = rowIdxFirst + numElements
        outputSize  = numElements
    elif convMode == ConvMode.VALID:
        rowIdxFirst = kernelLength - 1
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

    # If the kernel is [a, b, c] then the matrix (Full):
    # [a 0 0 0 0]
    # [b a 0 0 0]
    # [c b a 0 0]
    # [0 c b a 0]
    # [0 0 c b a]
    # [0 0 0 c b]
    # [0 0 0 0 c]
    # Looking at the columns, the kernel slides.
    
    for jj in range(jjMax):
        for ii in range(iiMax):
            # Building the matrix over the columns first
            if ((ii + jj >= rowIdxFirst) and (ii + jj < rowIdxLast)):
                # Valid output matrix row index
                vI[mtxIdx] = ii + jj - rowIdxFirst
                vJ[mtxIdx] = jj
                vV[mtxIdx] = vK[ii]
                mtxIdx    += 1
    
    
    # SciPy, like MATLAB is additive: mK[vI[k], vJ[k]] += vV[k]
    mK = sp.sparse.csr_matrix((vV, (vI, vJ)), shape = (outputSize, numCols))
    
    return mK


