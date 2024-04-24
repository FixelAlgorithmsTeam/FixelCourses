
# Python STD
from enum import auto, Enum, unique
# import math

# Data
import numpy as np
# import pandas as pd
import scipy as sp

# Machine Learning

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous

# Course Packages


# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# See https://docs.python.org/3/library/enum.html
@unique
class NNWeightInit(Enum):
    # Type of data in the CSV
    CONST   = auto()
    KAIMING = auto()
    XAVIER  = auto()


# Layers

# Linear Layer

class LinearLayer():
    def __init__( self, dimIn: int, dimOut: int, initMethod: NNWeightInit = NNWeightInit.KAIMING, initStd: float = 0.01 ) -> None:
        
        # Initialization
        if initMethod is NNWeightInit.CONST:
            weightsStd = initStd
        elif initMethod is NNWeightInit.KAIMING:
            weightsStd = np.sqrt(2 / dimIn)
        elif initMethod is NNWeightInit.XAVIER:
            weightsStd = np.sqrt(1 / dimIn)
        
        mW = weightsStd * np.random.randn(dimOut, dimIn)

        vB = np.zeros(dimOut)
        
        # Parameters
        self.mX      = None #<! Required for the backward pass
        self.dParams = {'mW' : mW,   'vB' : vB}
        self.dGrads  = {'mW' : None, 'vB' : None}
        
    def Forward( self, mX: np.ndarray ) -> np.ndarray:
        self.mX = mX #<! Required for the backward pass
        
        mW      = self.dParams['mW'] #<! Shape: (dimOut, dimIn)
        vB      = self.dParams['vB'] 
        mZ      = mW @ mX + vB[:, None]
        
        return mZ
    
    def Backward( self, mDz: np.ndarray ) -> np.ndarray:
        # Supports batch onf input by summing the gradients over each input.
        # Summing instead of averaging to support the case the loss is scaled by N.
        mW  = self.dParams['mW']
        mX  = self.mX
        
        vDb = np.sum(mDz, axis = 1) #<! Explicit Sum
        mDw = mDz @ mX.T #<! Implicit Sum
        mDx = mW.T @ mDz
        
        self.dGrads['vB'] = vDb
        self.dGrads['mW'] = mDw
                
        return mDx
    

# Activations

# ReLU Layer

class ReLULayer():
    def __init__( self ) -> None:
        
        self.mX = None #<! Required for the backward pass
        self.dParams = {}
        self.dGrads  = {}
    
    def Forward( self, mX: np.ndarray ) -> np.ndarray:
        
        self.mX = mX                 #-- store for Backward
        mZ      = np.maximum(mX, 0)
        
        return mZ
    
    def Backward( self, mDz: np.ndarray ) -> np.ndarray:
        
        mX  = self.mX
        mDx = np.where(mX > 0, mDz, 0.0)
                
        return mDx

# LeakyReLU Layer

class LeakyReLULayer():
    def __init__( self, α: float = 0.01 ) -> None:
        
        self.mX = None #<! Required for the backward pass
        self.α  = α
        self.dParams = {}
        self.dGrads  = {}
    
    def Forward( self, mX: np.ndarray ) -> np.ndarray:

        self.mX = mX                 #-- store for Backward
        mZ      = np.where(mX > 0, mX, self.α * mX)
        
        return mZ
    
    def Backward( self, mDz: np.ndarray ) -> np.ndarray:
        
        mX  = self.mX
        mDx = np.where(mX > 0, mDz, self.α * mDz)
                
        return mDx


# Loss Functions

# Cross Entropy Loss

def CrossEntropyLoss( vY: np.ndarray, mZ: np.ndarray ) -> Tuple[np.float_, np.ndarray]:
    '''
    Returns both the loss and the gradient w.r.t the input (mZ).
    Assumes the input is logits (Before applying probability like transformation).
    The function is equivalent of SoftMax + Cross Entropy.
    The function uses the mean loss (Normalized by N). 
    Hence gradients calculation should sum the gradients over the batch.
    '''
    N      = len(vY)
    mYHat   = sp.special.softmax(mZ, axis = 0)
    valLoss = -np.mean(np.log(mYHat[vY, range(N)]))
    
    mDz                = mYHat
    mDz[vY, range(N)] -= 1
    mDz               /= N #<! Now all needed is to sum gradients
    
    return valLoss, mDz

# MSE Loss

def MseLoss( vY: np.ndarray, vZ: np.ndarray ) -> Tuple[np.float_, np.ndarray]:
    '''
    Returns both the loss and the gradient w.r.t the input (vZ).
    The function uses the mean loss (Normalized by N). 
    Hence gradients calculation should sum the gradients over the batch.
    '''

    vDz = vZ - vY
    valLoss = np.mean(np.square(vDz))
    
    return valLoss, vDz


# Model Class

# NN Model
class ModelNN():
    def __init__( self, lLayers: List ) -> None:
        
        self.lLayers = lLayers
        
    def Forward( self, mX:np.ndarray ) -> np.ndarray:
        
        for oLayer in self.lLayers:
            mX = oLayer.Forward(mX)
        return mX
    
    def Backward( self, mDz: np.ndarray ) -> None:
        
        for oLayer in reversed(self.lLayers):
            mDz = oLayer.Backward(mDz)


# Training Loop

# Auxiliary Functions

def CountModelParams( oModel: ModelNN ) -> int:
    """
    Calculates the number of parameters of a model.  
    Input:
        oModel      - ModelNN which supports `Forward()` and `Backward()` methods.
    Output:
        numParams   - Scalar of the number of parameters in the model.
    Remarks:
      - AA
    """

    numParams = 0
    for oLayer in oModel.lLayers:
        for paramStr in oLayer.dParams: #<! Iterating on dictionary (Keys by default)
            numParams += np.size(oLayer.dParams[paramStr])
    
    return numParams