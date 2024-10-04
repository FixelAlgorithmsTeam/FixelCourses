
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
import matplotlib.pyplot as plt

# Miscellaneous
from enum import auto, Enum, unique

# Typing
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

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
    LINE_SEARCH = auto()

# Constants


# Optimization

class GradientDescent():
    def __init__( self, vX: np.ndarray, hGradFun: Callable, μ: float, /, *, stepSizeMode: StepSizeMode = StepSizeMode.CONSTANT, hObjFun: Callable = None, α: float = 0.5 ) -> None:
        
        dataDim = len(vX)
        
        self._dataDim       = dataDim
        self._hGradFun      = hGradFun
        self.μ              = μ #<! Step Size
        self._stepSizeMode  = stepSizeMode #<! Step Size Mode
        self._hObjFun       = hObjFun #<! Objective function
        self.α              = α #<! Backtracking constant
        self.K              = 20 #<! Maximum Backtracking iterations

        self.vX = np.copy(vX) #<! Current State
        self.vG = np.empty_like(vX) #<! Current Gradient
        self.vZ = np.empty_like(vX) #<! Buffer
        self.ii = 1
        
        pass

    # @njit
    def _ApplyIterationAdaptive( self ) -> np.ndarray:

        self.vG     = self._hGradFun(self.vX)
        currObjVal  = self._hObjFun(self.vX)
        self.vZ     = self.vX - self.μ * self.vG

        kk = 0
        while((self._hObjFun(self.vZ) > currObjVal) and (kk < self.K)):
            # For production code, must be limited by value of `self.μ` and number iterations
            self.μ *= self.α
            self.vZ = self.vX - self.μ * self.vG
            kk      += 1
        
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

class CoordinateDescent():
    def __init__( self, vX: np.ndarray, hGradFun: Callable, μ: float, /, *, stepSizeMode: StepSizeMode = StepSizeMode.CONSTANT, hObjFun: Callable = None, α: float = 0.5 ) -> None:
        
        dataDim = len(vX)
        
        self._dataDim       = dataDim
        self._hGradFun      = hGradFun #<! Gradient function (Coordinate)
        self.μ              = μ #<! Step Size
        self._stepSizeMode  = stepSizeMode #<! Step Size Mode
        self._hObjFun       = hObjFun #<! Objective function
        self.α              = α #<! Backtracking constant
        self.K              = 20 #<! Maximum Backtracking iterations

        self.vX = np.copy(vX) #<! Current State
        self.vZ = np.empty_like(vX) #<! Buffer
        self.ii = 1
        
        pass

    # @njit
    def _ApplyIterationAdaptive( self ) -> np.ndarray:

        self.vZ[:] = self.vX
        for jj in range(self._dataDim):
            valG        = self._hGradFun(self.vX, jj)
            currObjVal  = self._hObjFun(self.vX)
            self.vZ[jj] = self.vX[jj] - self.μ * valG
            
            kk = 0
            while((self._hObjFun(self.vZ) > currObjVal) and (kk < self.K)):
                # For production code, must be limited by value of `self.μ` and number iterations
                self.μ     *= self.α
                self.vZ[jj] = self.vX[jj] - self.μ * valG
                kk         += 1
            
            self.μ       = max(1e-9, self.μ)
            self.vX[jj] -= self.μ * valG
            self.μ      /= self.α

        self.ii += 1

        return self.vX
    
    # @njit
    def ApplyIteration( self ) -> np.ndarray:

        if (self._stepSizeMode == StepSizeMode.ADAPTIVE):
            return self._ApplyIterationAdaptive()
        
        for jj in range(self._dataDim):
            valG         = self._hGradFun(self.vX, jj)
            self.vX[jj] -= self.μ * valG

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
        """
        Solves F(X) = f(x) + λ * g(x)
        Where the Gradient of f is known and the prox of g is known.

        hProxFun(vY, λ) -> Prox
        """
        
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


def ProjectSimplexBall( vY: np.ndarray, /, *, ballRadius: float = 1.0, ε: float = 0.0 ) -> np.ndarray:
    """
    Solving the Orthogonal Projection Problem of the input vector onto the 
    Simplex Ball using Dual Function and exact solution by solving linear 
    equation.
    Input:
    - vY            -   Input Vector.
                        Structure: Vector.
                        Type: 'Float'.
                        Range: (-inf, inf).
    - ballRadius    -   Ball Radius.
                        Sets the Radius of the Simplex Ball. For Unit
                        Simplex set to 1.
                        Structure: Scalar.
                        Type: 'Float'.
                        Range: (0, inf).
    Output:
    - vX            -   Output Vector.
                        The projection of the Input Vector onto the Simplex
                        Ball.
                        Structure: Vector.
                        Type: 'Float'.
                        Range: (-inf, inf).
    References
    1.  A
    Remarks:
    1.  The solver finds 2 points which one is positive and the other is
        negative. Then, since the objective function is linear, finds the
        exact point where the linear function has value of zero.
    TODO:
      1.  U.
    Release Notes:
      -   1.0.000     02/10/2024  Royi Avital
          *   First release version.
    """
    
    if ((np.fabs((np.sum(vY) - ballRadius)) <= ε) and (np.all(vY >= 0))):
        # The input is already within the Ball.
        vX = np.copy(vY)
        return vX
    
    vZ = np.sort(vY)
    
    vλ    = np.r_[vZ[0] - ballRadius, vZ, vZ[-1] + ballRadius] #<! The range guarantees at least one positive and one negative value
    hObjFun = lambda λ: np.sum( np.maximum(vY - λ, 0) ) - ballRadius
    
    vObjVal = np.zeros_like(vλ)
    
    for ii, valλ in enumerate(vλ):
        vObjVal[ii] = hObjFun(valλ)
    
    if (np.any(vObjVal == 0)):
        λ = vλ[vObjVal == 0][0] #<! In case more than a single value gets zero
    else:
        # Working on when an Affine Function have the value zero
        valX1Idx = np.flatnonzero(vObjVal > 0)[-1]
        valX2Idx = np.flatnonzero(vObjVal < 0)[0]
        
        valX1 = vλ[valX1Idx]
        valX2 = vλ[valX2Idx]
        valY1 = vObjVal[valX1Idx]
        valY2 = vObjVal[valX2Idx]
        
        paramA      = (valY2 - valY1) / (valX2 - valX1)
        paramB      = valY1 - (paramA * valX1)
        λ = -paramB / paramA
        
    vX = np.maximum(vY - λ, 0)

    return vX

def ProjectL1Ball( vY: np.ndarray, /, *, ballRadius: float = 1.0, ε: float = 0.0 ) -> np.ndarray:
    """
    Solving the Orthogonal Projection Problem of the input vector onto the L1 
    Ball using Dual Function and exact solution by solving linear equation.
    Input:
    - vY            -   Input Vector.
                        Structure: Vector.
                        Type: 'Float'.
                        Range: (-inf, inf).
    - ballRadius    -   Ball Radius.
                        Sets the Radius of the Simplex Ball. For Unit
                        Simplex set to 1.
                        Structure: Scalar.
                        Type: 'Float'.
                        Range: (0, inf).
    Output:
    - vX            -   Output Vector.
                        The projection of the Input Vector onto the Simplex
                        Ball.
                        Structure: Vector.
                        Type: 'Float'.
                        Range: (-inf, inf).
    References
    1.  A
    Remarks:
    1.  The solver finds 2 points which one is positive and the other is
        negative. Then, since the objective function is linear, finds the
        exact point where the linear function has value of zero.
    TODO:
      1.  U.
    Release Notes:
      -   1.0.000     02/10/2024  Royi Avital
          *   First release version.
    """
    
    if ((np.linalg.norm(vY, 1) - ballRadius) <= ε):
        # The input is already within the L1 Ball.
        vX = np.copy(vY)
        return vX
    
    vZ = np.sort(np.abs(vY))
    
    vλ    = np.r_[0, vZ, vZ[-1] + ballRadius] #<! The range guarantees at least one positive and one negative value
    hObjFun = lambda λ: np.sum( np.maximum(vZ - λ, 0) ) - ballRadius
    
    vObjVal = np.zeros_like(vλ)
    
    for ii, valλ in enumerate(vλ):
        vObjVal[ii] = hObjFun(valλ)
    
    if (np.any(vObjVal == 0)):
        λ = vλ[vObjVal == 0][0] #<! In case more than a single value gets zero
    else:
        # Working on when an Affine Function have the value zero
        valX1Idx = np.flatnonzero(vObjVal > 0)[-1]
        valX2Idx = np.flatnonzero(vObjVal < 0)[0]
        
        valX1 = vλ[valX1Idx]
        valX2 = vλ[valX2Idx]
        valY1 = vObjVal[valX1Idx]
        valY2 = vObjVal[valX2Idx]
        
        paramA      = (valY2 - valY1) / (valX2 - valX1)
        paramB      = valY1 - (paramA * valX1)
        λ = -paramB / paramA
        
    vX = np.sign(vY) * np.maximum(np.fabs(vY) - λ, 0)

    return vX

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

# Data

def MakeSignal( signalType: Literal['Blocks', 'Bumps', 'Chirps', 'Cusp', 'Cusp2', 
                                    'Doppler', 'Gabor', 'Gaussian', 'HeaviSine', 'HiSine', 
                                    'HypChirps', 'Leopold', 'LinChirp', 'LinChirps', 'LoSine', 
                                    'MishMash', 'Piece-Polynomial', 'Piece-Regular', 'QuadChirp', 
                                    'Ramp', 'Riemann', 'SineOneOverX', 'Sing', 'SmoothCusp', 'TwoChirp', 
                                    'WernerSorrows'], numSamples: int ) -> np.ndarray:
    """
    MakeSignal -- Make artificial signal
    Usage: `sig = MakeSignal(signalType, numSamples)`
    
    Inputs
        - signalType: string, type of signal to generate
        - numSamples: int, desired signal length
    
    Outputs
        - sig: 1D signal (numpy array)
    
    Remarks:
     - 'Leopold': Kronecker.
     - 'Piece-Polynomial': Piece Wise 3rd degree polynomial.
     - 'Piece-Regular': Piece Wise Smooth.
     - Code with assistance of ClaudeAI and ChatGPT.
    References
     - Various articles of D.L. Donoho and I.M. Johnstone.
    """

    vT = np.arange(1, numSamples + 1) / numSamples
    if signalType == 'Blocks':
        pos = [0.10,  0.13,  0.15,  0.23,  0.25,  0.40,  0.44, 0.65,  0.76,  0.78,  0.81]
        hgt = [4.00, -5.00,  3.00, -4.00,  5.00, -4.20,  2.10, 4.30, -3.10,  2.10, -4.20]
        vS = np.zeros_like(vT)
        for jj in range(len(pos)):
            vS += (1 + np.sign(vT - pos[jj])) * (hgt[jj] / 2 )
    elif signalType == 'Bumps':
        pos = [0.10,  0.13,  0.15,  0.23,  0.25,  0.40,  0.44, 0.65,  0.76,  0.78,  0.81]
        hgt = [4.00,  5.00,  3.00,  4.00,  5.00,  4.20,  2.10, 4.30,  3.10,  5.10,  4.20]
        wth = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005]
        vS = np.zeros_like(vT)
        for jj in range(len(pos)):
            vS += hgt[jj] / np.power(1 + np.abs((vT - pos[jj]) / wth[jj]), 4)
    elif signalType == 'Chirps':
        t = vT * 10 * np.pi
        f1 = np.cos(t**2 * numSamples / 1024)
        a = 30 * numSamples / 1024
        t = vT * np.pi
        f2 = np.cos(a * (t**3))
        f2 = f2[::-1]  # reverse
        ix = 20 * np.linspace(-numSamples, numSamples, 2 * numSamples + 1) / numSamples
        g = np.exp(-np.square(ix) * 4 * numSamples / 1024)
        i1 = slice(numSamples // 2, numSamples // 2 + numSamples)
        i2 = slice(numSamples // 8, numSamples // 8 + numSamples)
        j = vT
        f3 = g[i1] * np.cos(50 * np.pi * j * numSamples / 1024)
        f4 = g[i2] * np.cos(350 * np.pi * j * numSamples / 1024)
        vS = f1 + f2 + f3 + f4
        envelope = np.ones(numSamples)
        envelope[:numSamples // 8] = (1 + np.sin(-np.pi/2 + np.linspace(0, np.pi, numSamples//8))) / 2
        envelope[7*numSamples//8:] = envelope[numSamples//8-1::-1]
        vS = vS * envelope
    elif signalType == 'Cusp':
        vS = np.sqrt(np.abs(vT - 0.37))
    elif signalType == 'Cusp2':
        N = 64
        i1 = np.arange(1, N + 1) / N
        x = (1 - np.sqrt(i1)) + (i1 / 2) - 0.5
        M = 8 * N
        vS = np.zeros(M)
        vS[int(M - 1.5*N):int(M - 0.5*N)] = x
        vS[int(M - 2.5*N + 1):int(M - 1.5*N + 1)] = x[::-1]
        vS[3*N:4*N] = 0.5 * np.ones(N)
    elif signalType == 'Doppler':
        vS = np.sqrt(vT * (1 - vT)) * np.sin((2 * np.pi * 1.05) / (vT + 0.05))
    
    
    return vS

# Visualization

def DisplayRunSummary( solverName: str, hObjFun: Callable, vX: np.ndarray, runTime: float, cvxpyStatus: Optional[bool] = None ) -> None:

    print('')
    print(f'{solverName} Solution Summary:' )
    if cvxpyStatus is not None:
        print(f' - The {solverName} Solver Status         : {cvxpyStatus}')
    
    print(f' - The Optimal Value Is Given By   : {hObjFun(vX)}')
    print(f' - The Optimal Argument Is Given By: {np.array_str(vX, max_line_width = np.inf)}') #<! https://stackoverflow.com/a/49437904
    print(f' - The Run Time Is Given By        : {runTime:0.3f} [Sec]')
    print(' ')

    return

def DisplayCompaisonSummary( dSolverData: Dict[str, Dict], hObjFun: Callable, /, *, figSize: Tuple[int, int] = (12, 9), refSolverName: str = 'CVXPY', ε: float = 1e-8 ) -> plt.Figure:

    refSolver  = False
    numSolvers = len(dSolverData)
    
    if refSolverName in dSolverData.keys():
        vXRef       = dSolverData[refSolverName]['vX']
        xNormRef    = max(np.linalg.norm(vXRef), ε)
        objValRef   = max(dSolverData[refSolverName]['objVal'], ε)

        refSolver   = True
        numSolvers -= 1 #<! Compare solvers to reference

    if refSolver:
        # Show the objective value over the iterations
        hF, vHA = plt.subplots(nrows = 2, ncols = 1, figsize = figSize)

        for solName, dSolData in dSolverData.items():
            if solName == refSolverName:
                continue


            mX = dSolData['mX']
            lObjErr = [20 * np.log10(max(abs(hObjFun(mX[:, ii]) - objValRef), ε) / objValRef) for ii in range(np.size(mX, 1))]
            lArgErr = [20 * np.log10(max(np.linalg.norm(mX[:, ii] - vXRef), ε) / xNormRef) for ii in range(np.size(mX, 1))]

            hA = vHA.flat[0] #<! Objective Value
            hA.plot(lObjErr, lw = 2, label = solName)
            # hA.set_xlabel('Iteration Index') $<! No need, shared with the one below
            hA.set_ylabel('Relative Error [dB]')
            hA.set_title(f'Objective Value of the Solvers Compared to {refSolverName}')
            hA.legend()

            hA = vHA.flat[1] #<! Objective Value
            hA.plot(lArgErr, lw = 2, label = solName)
            hA.set_xlabel('Iteration Index')
            hA.set_ylabel('Relative Error [dB]')
            hA.set_title(f'Argument of the Solvers Compared to {refSolverName}')
            hA.legend()

    else:
        # Show the objective value over the iterations
        hF, hA = plt.subplots(figsize = figSize)

        for solName, dSolData in dSolverData.items():

            mX = dSolData['mX']
            lObjVal = [hObjFun(mX[:, ii]) for ii in range(np.size(mX, 1))]

            hA.plot(lObjVal, lw = 2, label = solName)
        
        hA.legend()
        hA.set_xlabel('Iteration Index')
        hA.set_ylabel('Objective Value')
        hA.set_title('Objective Value of the Solvers')
        hA.legend()

    return hF
    


