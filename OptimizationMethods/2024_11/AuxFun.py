
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
    """
    Gradient Descent Solver with Constant or Adaptive Step Size Option.

    This class implements the standard gradient descent algorithm with support for adaptive step size 
    adjustment (Backtracking line search) based on the objective function value, or a constant step size mode.

    Parameters:
    -----------
    vX : np.ndarray
        Initial point (starting state) of the optimization.
        Structure: Vector (1D array).
        Type: `float` or `double`.

    hGradFun : Callable
        Function that computes the gradient of the objective function `f(x)`. This should take the current 
        iterate `vX` as input and return the gradient at that point.
        Type: `Callable[[np.ndarray], np.ndarray]`.

    μ : float
        Initial step size (Learning Rate) for the gradient descent update. It can be fixed or adjusted adaptively 
        based on the chosen `stepSizeMode`.

    stepSizeMode : StepSizeMode, optional
        The mode for step size control. Options are:
        - `StepSizeMode.CONSTANT`: The step size remains constant at its initial value `μ` throughout the iterations.
        - `StepSizeMode.ADAPTIVE`: The step size is adjusted using backtracking line search, ensuring sufficient 
          decrease in the objective function.
        Default: `StepSizeMode.CONSTANT`.

    hObjFun : Callable, optional
        Objective function `f(x)`, used in the case of adaptive step size. If `stepSizeMode` is `ADAPTIVE`, 
        this function is required to evaluate the current value of `f(x)` for backtracking. If not provided, 
        the constant step size mode will be used.
        Type: `Optional[Callable[[np.ndarray], float]]`.
        Default: `None`.

    α : float, optional
        The reduction factor for step size in each iteration of backtracking. Must be in the range (0, 1).
        Only used when `stepSizeMode` is `ADAPTIVE`.
        Default: `0.5`.

    Attributes:
    -----------
    vX : np.ndarray
        Current state (iterate) of the optimization process.
    
    vG : np.ndarray
        Latest calculation gradient.

    ii : int
        Iteration counter, tracks the current iteration number.

    μ : float
        The current step size used in the gradient descent update, which may change if adaptive mode is used.

    α : float
        Backtracking line search constant used for reducing step size in adaptive mode.

    K : int
        Maximum number of backtracking iterations allowed for step size reduction.

    Methods:
    --------
    ApplyIteration() -> np.ndarray:
        Applies one iteration of the gradient descent algorithm. Depending on the step size mode, it either 
        performs a constant step update or an adaptive update with backtracking line search.

    ApplyIterations(numIterations: int, *, logArg: bool = True) -> Optional[List[np.ndarray]]:
        Applies a specified number of iterations of the gradient descent algorithm. Optionally logs 
        the intermediate states.

    Notes:
    ------
    - The adaptive step size mode performs backtracking line search to ensure sufficient decrease in the objective 
      function. This is useful for cases where the objective function has sharp gradients or is not well-behaved 
      with a fixed step size.
    - If adaptive mode is used, it is essential to provide a valid objective function `hObjFun` to evaluate the 
      objective values during backtracking.

    References:
    -----------
    1. Nocedal, J., & Wright, S. J. (2006). Numerical Optimization. Springer Science & Business Media.
    2. Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
    """

    def __init__( self, vX: np.ndarray, hGradFun: Callable[[np.ndarray], np.ndarray], μ: float, /, *, stepSizeMode: StepSizeMode = StepSizeMode.CONSTANT, hObjFun: Optional[Callable[[np.ndarray], float]] = None, α: float = 0.5 ) -> None:
        
        dataDim = len(vX)

        if ((stepSizeMode == StepSizeMode.ADAPTIVE) and (hObjFun is None)):
            raise ValueError(f'If `stepSizeMode` is ste to adaptive, an objective function must be supplied')
        
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
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List[np.ndarray]]:

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
    """
    Coordinate Descent Solver with Adaptive Step Size Option.

    This class implements the coordinate descent algorithm, where optimization is performed one coordinate at a time, 
    allowing for flexible step size control (constant or adaptive). The algorithm adjusts each coordinate based on the 
    gradient with respect to that coordinate, making it suitable for sparse optimization problems or when the 
    objective function is separable.

    Parameters:
    -----------
    vX : np.ndarray
        Initial point (starting state) of the optimization. This is the initial guess for the solution.
        Structure: Vector (1D array).
        Type: `float` or `double`.

    hGradFun : Callable[[np.ndarray], np.ndarray]
        Function that computes the gradient of the objective function `f(x)` with respect to a single coordinate.
        This function should take two inputs: the current iterate `vX` and the index of the coordinate, and return 
        the gradient at that coordinate.
        Type: `Callable[[np.ndarray, int], float]`.

    μ : float
        Initial step size (Learning Rate) for the coordinate-wise gradient update. It can be fixed or adjusted adaptively 
        depending on the chosen `stepSizeMode`.

    stepSizeMode : StepSizeMode, optional
        The mode for step size control. Options are:
        - `StepSizeMode.CONSTANT`: The step size remains constant at its initial value `μ` throughout the iterations.
        - `StepSizeMode.ADAPTIVE`: The step size is adjusted using backtracking line search based on the objective 
          function values, ensuring sufficient decrease in the objective function.
        Default: `StepSizeMode.CONSTANT`.

    hObjFun : Callable[[np.ndarray], float], optional
        Objective function `f(x)`, used in the case of adaptive step size. If `stepSizeMode` is set to `ADAPTIVE`, 
        this function is required to evaluate the current value of `f(x)` for backtracking. If not provided, 
        the constant step size mode will be used.
        Type: `Callable[[np.ndarray], float]`.
        Default: `None`.

    α : float, optional
        Reduction factor for step size during backtracking. Must be in the range (0, 1). This parameter is used 
        only when `stepSizeMode` is `ADAPTIVE`.
        Default: `0.5`.

    Attributes:
    -----------
    vX : np.ndarray
        Current state (iterate) of the optimization process.
    
    vZ : np.ndarray
        Buffer used to store updated coordinate values during each iteration.

    ii : int
        Iteration counter, which tracks the current iteration number.

    μ : float
        The current step size used in the coordinate-wise gradient update, which may change if adaptive mode is used.

    α : float
        Backtracking line search constant used for reducing the step size in adaptive mode.

    K : int
        Maximum number of backtracking iterations allowed for step size reduction in adaptive mode.

    Methods:
    --------
    ApplyIteration() -> np.ndarray:
        Applies one iteration of the coordinate descent algorithm. Depending on the step size mode, it either 
        performs a constant step update or an adaptive update with backtracking line search for each coordinate.

    ApplyIterations(numIterations: int, *, logArg: bool = True) -> Optional[List[np.ndarray]]:
        Applies a specified number of iterations of the coordinate descent algorithm. Optionally logs the 
        intermediate states.

    Notes:
    ------
    - The adaptive step size mode performs backtracking line search to ensure sufficient decrease in the objective 
      function. This is useful in cases where a constant step size may not be optimal, such as when the objective 
      function has sharp gradients or poorly conditioned coordinates.
    - If adaptive mode is used, it is essential to provide a valid objective function `hObjFun()` to evaluate the 
      objective values during backtracking.
    - Coordinate Descent is often more efficient for problems where the objective function is separable or when 
      gradient computation is costly.

    References:
    -----------
    1. Nesterov, Y. (2012). Efficiency of coordinate descent methods on huge-scale optimization problems. 
       SIAM Journal on Optimization, 22(2), 341-362.
    2. Tseng, P., & Yun, S. (2009). A coordinate gradient descent method for nonsmooth separable minimization. 
       Mathematical Programming, 117(1), 387-423.
    """

    def __init__( self, vX: np.ndarray, hGradFun: Callable[[np.ndarray], np.ndarray], μ: float, /, *, stepSizeMode: StepSizeMode = StepSizeMode.CONSTANT, hObjFun: Optional[Callable[[np.ndarray], float]] = None, α: float = 0.5 ) -> None:
        
        dataDim = len(vX)

        if ((stepSizeMode == StepSizeMode.ADAPTIVE) and (hObjFun is None)):
            raise ValueError(f'If `stepSizeMode` is ste to adaptive, an objective function must be supplied')
        
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
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List[np.ndarray]]:

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
    """
    Proximal Gradient Descent Solver for Composite Objective Functions.

    This class solves composite objective functions of the form:

        F(x) = f(x) + λ * g(x)

    where `f(x)` is a smooth function with a known gradient and `g(x)` is a function with a known proximal operator (Usually non smooth). 
    The proximal gradient descent method is used to minimize this objective function, and an accelerated version (FISTA style) 
    is optionally available for faster convergence.

    Parameters:
    -----------
    vX : np.ndarray
        Initial point (starting state) of the optimization.
        Structure: Vector (1D array).
        Type: `float` or `double`.

    hGradFun : Callable
        Function that computes the gradient of `f(x)`. This should take the current iterate `vX` as input 
        and return the gradient at that point.
        Structure: Function.
        Type: `Callable[[np.ndarray], np.ndarray]`.

    μ : float
        Step size (learning rate) used in the gradient descent update. It controls how large the step in 
        the direction of the gradient should be.
        Range: (0, ∞).

    λ : float
        Regularization parameter controlling the weight of the non-smooth term `g(x)` in the objective function.
        Range: (0, ∞).

    hProxFun : Callable, optional
        The proximal operator of the non-smooth function `g(x)`. It takes the argument `vY` and returns 
        the proximal map of `g` with respect to `λ`. By default, the identity function is used (i.e., no prox applied).
        Structure: Function.
        Type: `Callable[[np.ndarray, float], np.ndarray]`.
        Default: `lambda vX, λ: vX` (No prox function).

    useAccel : bool, optional
        If True, the accelerated proximal gradient descent algorithm (FISTA) is used, otherwise 
        standard proximal gradient descent is applied.
        Default: `False`.

    Attributes:
    -----------
    vX : np.ndarray
        Current state (iterate) of the optimization process.
    
    vG : np.ndarray
        Current iteration gradient at the point used for calculation.

    vV : np.ndarray
        Buffer vector used for momentum acceleration in FISTA.

    ii : int
        Iteration counter, tracks the current iteration number.

    useAccel : bool
        Indicates whether the FISTA acceleration is enabled (`True`) or standard gradient descent is used (`False`).

    Methods:
    --------
    ApplyIteration() -> np.ndarray:
        Applies one iteration of the proximal gradient descent algorithm. If acceleration is enabled, FISTA is used.

    ApplyIterations(numIterations: int, *, logArg: bool = True) -> Optional[List[np.ndarray]]:
        Applies a specified number of iterations of the proximal gradient descent algorithm. Optionally logs 
        the intermediate states.

    Notes:
    ------
    - The accelerated method implemented here follows the FISTA algorithm, where momentum is applied using a linear 
      combination of previous iterates.
    - The `ApplyIterations()` method allows the user to perform multiple iterations at once and optionally store 
      the intermediate results for further analysis.

    References:
    -----------
    1. Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage Thresholding Algorithm for Linear Inverse Problems. 
       SIAM Journal on Imaging Sciences, 2(1), 183-202.
    2. Parikh, N., & Boyd, S. (2014). Proximal Algorithms. Foundations and Trends® in Optimization, 1(3), 127-239.
    """

    def __init__( self, vX: np.ndarray, hGradFun: Callable[[np.ndarray], np.ndarray], μ: float, λ: float, /, *, hProxFun: Callable[[np.ndarray, float], np.ndarray] = lambda vX, λ: vX, useAccel: bool = False ) -> None:
        
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
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List[np.ndarray]]:

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

class ADMM():
    """
    Alternating Direction Method of Multipliers (ADMM) for Solving Composite Optimization Problems.

    This class implements the ADMM algorithm to solve problems of the form:

        F(x) = f(x) + λ * g(P * x)    ->    f(x) + λ * g(z), subject to P * x - z = 0

    Where:
    - `f(x)` is a function that is easily minimized (e.g., least squares).
    - `g(z)` is a function that has an efficient proximal mapping.
    - `P` is a linear operator (matrix) that transforms `x` into `z`.

    ADMM splits the problem into subproblems that alternately minimize over `x` and `z`, with a dual update for the constraint violation.  
    This method is well suited for large scale or distributed optimization problems where the objective is separable or involves a linear constraint.

    Parameters:
    -----------
    vX : np.ndarray
        Initial point (starting state) for the variable `x`.
        Structure: Vector (1D array).
        Type: `float` or `double`.

    hMinFun : Callable[[np.ndarray, np.ndarray, float], np.ndarray]
        A function that minimizes the function `f(x)` and a quadratic penalty term `(ρ / 2) * || P * x - z + w ||_2^2`.
        This function should take three arguments: the current values of `z`, the dual variable `w`, and the penalty parameter `ρ`, 
        and return the next `x` value that minimizes `f(x) + (ρ / 2) * || P * x - z + w ||_2^2`.

    hProxFun : Callable[[np.ndarray, float], np.ndarray]
        A function that computes the proximal mapping of `g(z)` with respect to the current value of `P * x + w`.
        This function should take two inputs: the current value of `P * x + w`, and the scaled penalty parameter `λ / ρ`, 
        and return the next `z` value that minimizes `λ * g(z) + (ρ / 2) * || P * x - z + w ||_2^2`.

    mP : np.ndarray
        The linear operator (matrix) that transforms the variable `x` into `z`, i.e., the matrix `P` in the constraint `Px - z = 0`.

    ρ : float, optional
        Penalty parameter for the augmented Lagrangian. This parameter controls the tradeoff between the primal and dual residuals 
        in the optimization. Larger values of `ρ` put more emphasis on satisfying the constraint `P * x - z = 0`.
        Range: (0, inf).
        Default: `1.0`.

    λ : float, optional
        The regularization parameter that weights the term `g(z)` in the objective function. 
        Range: (0, inf).
        Default: `1.0`.

    Attributes:
    -----------
    vX : np.ndarray
        Current state (iterate) of the variable `x`.
    
    vZ : np.ndarray
        Current state (iterate) of the variable `z`, introduced to split the problem.

    vW : np.ndarray
        Dual variable (Lagrangian multiplier) associated with the constraint `P * x - z = 0`. 
        This is updated in each iteration to ensure convergence.

    ρ : float
        The current value of the penalty parameter.

    λ : float
        The regularization parameter for the objective function.

    ii : int
        Iteration counter, tracking the current number of iterations.

    Methods:
    --------
    ApplyIteration() -> np.ndarray:
        Performs one iteration of the ADMM algorithm, which consists of the following steps:
        1. Minimize with respect to `x` by solving the subproblem for `f(x)`.
        2. Minimize with respect to `z` using the proximal mapping for `g(z)`.
        3. Update the dual variable `w`.

    ApplyIterations(numIterations: int, *, logArg: bool = True) -> Optional[List[np.ndarray]]:
        Applies a specified number of iterations of the ADMM algorithm. Optionally logs the intermediate states for 
        each iteration.
        Parameters:
        - `numIterations`: The number of ADMM iterations to run.
        - `logArg`: Whether to store and return the intermediate values of `x` during the iterations.
          If `True`, a list of all intermediate `x` values will be returned.
          Default: `True`.

    Notes:
    ------
    - ADMM is an efficient and flexible method for solving optimization problems that involve separable objectives and 
      linear constraints.
    - The algorithm alternates between minimizing `x` and `z`, which allows for parallelization or distributed 
      computation in some applications.
    - Convergence is typically achieved when the primal and dual residuals (measures of constraint violation and dual 
      update) become sufficiently small.

    References:
    -----------
    1. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends® in Machine Learning, 3(1), 1-122.
    2. Gabay, D., & Mercier, B. (1976). A dual algorithm for the solution of nonlinear variational problems via finite element approximation. Computers & Mathematics with Applications, 2(1), 17-40.
    """

    def __init__( self, vX: np.ndarray, hMinFun: Callable[[np.ndarray, np.ndarray, float], np.ndarray], hProxFun: Callable[[np.ndarray, float], np.ndarray], mP: np.ndarray, /, *, ρ: float = 1.0, λ: float = 1.0 ) -> None:
        """
        Solves the model: F(x) = f(x) + λ g(P * x) -> f(x) + λ g(z), subject to Px - z = 0
        Where:
         - f(x) + Least Squares is easy to solve.
         - g(z) has efficient Proximal Mapping.
        """
        
        dataDim = len(vX)
        
        self._dataDim = dataDim
        self._hMinFun = hMinFun   #<! Solves \arg \min_x f(x) + (ρ / 2) * || P * x - z + w ||_2^2
        self._hProxFun = hProxFun #<! Solves \arg \min_z λ * g(z) + (ρ / 2) * || P * x - z + w ||_2^2
        self._mP = mP #<! Linear Model Px - z = 0
        self.ρ = ρ #<! Penalty coefficient
        self.λ = λ #<! Parameter Lambda

        self.vX  = np.copy(vX) #<! Current State
        self.vZ  = np.empty(np.size(mP, 0)) #<! Separable variable
        self.vW  = np.empty(np.size(mP, 0)) #<! Dual Variable (Lagrangian Multiplier)
        self.ii  = 1
        
        pass
    
    # @njit
    def ApplyIteration( self ) -> np.ndarray:
        
        self.vX  = self._hMinFun(self.vZ, self.vW, self.ρ) #<! Minimizing for x
        self.vZ  = self._hProxFun(self._mP @ self.vX + self.vW, self.λ / self.ρ) #<! Minimizing for z
        self.vW += self._mP @ self.vX - self.vZ #<! Dual variable update

        self.ii += 1

        return self.vX
    
    def ApplyIterations( self, numIterations: int, *, logArg: bool = True ) -> Optional[List[np.ndarray]]:

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

def DisplayRunSummary( solverName: str, hObjFun: Callable[[np.ndarray], float], vX: np.ndarray, runTime: float, cvxpyStatus: Optional[bool] = None ) -> None:

    print('')
    print(f'{solverName} Solution Summary:' )
    if cvxpyStatus is not None:
        print(f' - The {solverName} Solver Status         : {cvxpyStatus}')
    
    print(f' - The Optimal Value Is Given By   : {hObjFun(vX)}')
    print(f' - The Optimal Argument Is Given By: {np.array_str(vX, max_line_width = np.inf)}') #<! https://stackoverflow.com/a/49437904
    print(f' - The Run Time Is Given By        : {runTime:0.3f} [Sec]')
    print(' ')

    return

def DisplayCompaisonSummary( dSolverData: Dict[str, Dict], hObjFun: Callable[[np.ndarray], float], /, *, figSize: Tuple[int, int] = (12, 9), refSolverName: str = 'CVXPY', ε: float = 1e-8 ) -> plt.Figure:

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
            # lObjErr = [20 * np.log10(max(abs(hObjFun(mX[ii]) - objValRef), ε) / objValRef) for ii in range(np.size(mX, 0))]
            # lArgErr = [20 * np.log10(max(np.linalg.norm(mX[ii] - vXRef), ε) / xNormRef) for ii in range(np.size(mX, 0))]

            lObjErr = 20 * np.log10(np.maximum(np.abs(np.apply_along_axis(hObjFun, 1, mX) - objValRef), ε) / objValRef)
            lArgErr = 20 * np.log10(np.maximum(np.linalg.norm(mX - vXRef[None, :], axis = 1), ε) / xNormRef)

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
    


