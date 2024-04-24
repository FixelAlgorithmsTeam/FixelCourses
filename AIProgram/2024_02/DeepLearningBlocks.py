
# Python STD
from enum import auto, Enum, unique
import math

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
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union

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

# Sequential NN Model
class ModelSequentialNN():
    def __init__( self, lLayers: List ) -> None:
        
        self.lLayers = lLayers
        
    def Forward( self, mX: np.ndarray ) -> np.ndarray:
        
        for oLayer in self.lLayers:
            mX = oLayer.Forward(mX)
        return mX
    
    def Backward( self, mDz: np.ndarray ) -> None:
        
        for oLayer in reversed(self.lLayers):
            mDz = oLayer.Backward(mDz)

# Union of Types
ModelNN = Union[ModelSequentialNN]

# Data

class DataSet():
    def __init__( self, mX: np.ndarray, vY: np.ndarray, batchSize: int, shuffleData: bool = True, dropLast: bool = True ) -> None:

        numSamples = len(vY)
        
        if batchSize > numSamples:
            raise ValueError(f'The batch size: {batchSize} is greater than the number of samples: {numSamples}')
        
        self.mX          = mX
        self.vY          = vY
        self.batchSize   = batchSize
        self.shuffleData = shuffleData #<! If shuffleData = False and numSamples / batchSize /= Int some samples will always left out
        self.numSamples  = numSamples
        if dropLast:
            self.numBatches  = numSamples // batchSize
        else:
            self.numBatches  = math.ceil(numSamples / batchSize)
        self.vIdx        = np.random.permutation(self.numSamples)
        
    def __len__( self ) -> int:
        
        return self.numBatches
    
    def __iter__( self ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        if self.shuffleData:
            vIdx = np.random.permutation(self.numSamples)
        else:
            vIdx = self.vIdx

        for ii in range(self.numBatches):
            startIdx  = ii       * self.batchSize
            endIdx    = min(startIdx + self.batchSize, self.numSamples) #<! Will work without the "safety net": lA = [None] * 3; lA[1:20]
            vBatchIdx = vIdx[startIdx:endIdx]
            mXBatch   = self.mX[:, vBatchIdx]
            vYBatch   = self.vY[vBatchIdx]

            yield mXBatch, vYBatch

# Training Loop

def TrainEpoch( oModel: ModelNN, oDataSet: DataSet, learnRate: float, hL: Callable, hS: Callable ) -> Tuple[float, float]:
    """
    Applies a single Epoch training of a model.  
    Input:
        oModel      - ModelNN object which supports `Forward()` and `Backward()` methods.
        oDataSet    - DataSet object which supports iterating.
        learnRate   - Scalar of the learning rate in the range (0, inf).
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
    """

    epochLoss   = 0.0
    epochScore  = 0.0
    numSamples  = 0
    for ii, (mX, vY) in enumerate(oDataSet):
        batchSize       = len(vY)
        # Forward
        mZ              = oModel.Forward(mX)
        valLoss, mDz    = hL(vY, mZ)
        
        # Backward
        oModel.Backward(mDz)
        
        # Gradient Descent (Update parameters
        for oLayer in oModel.lLayers:
            for sParam in oLayer.dGrads:
                oLayer.dParams[sParam] -= learnRate * oLayer.dGrads[sParam]
        
        # Score
        valScore = hS(mZ, vY)

        # Normalize so each sample has the same weight
        epochLoss  += batchSize * valLoss
        epochScore += batchSize * valScore
        numSamples += batchSize
    
            
    return epochLoss / numSamples, epochScore / numSamples


def ScoreEpoch( oModel: ModelNN, oDataSet: DataSet, hL: Callable, hS: Callable ) -> Tuple[float, float]:
    """
    Calculates the loss and the score of a model over an Epoch.  
    Input:
        oModel      - ModelNN which supports `Forward()` and `Backward()` methods.
        oDataSet    - DataSet object which supports iterating.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
    Remarks:
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The function does not optimize the model parameter.
    """
    
    epochLoss   = 0.0
    epochScore  = 0.0
    numSamples  = 0
    for ii, (mX, vY) in enumerate(oDataSet):
        batchSize       = len(vY)
        # Forward
        mZ              = oModel.Forward(mX)
        valLoss, mDz    = hL(vY, mZ)
        
        # Score
        valScore = hS(mZ, vY)

        epochLoss  += batchSize * valLoss
        epochScore += batchSize * valScore
        numSamples += batchSize
    
            
    return epochLoss / numSamples, epochScore / numSamples

# Score Functions

def ScoreAccLogits( mScore: np.ndarray, vY: np.ndarray ) -> np.float_:
    """
    Calculates the classification accuracy.  
    Input:
        mScore      - Matrix (numCls, batchSize) of the Logits Score.
        vY          - Vector (batchSize, ) of the reference classes: {0, 1, .., numCls - 1}.
    Output:
        valAcc      - Scalar of the accuracy in [0, 1] range.
    Remarks:
      - The Logits are assumed to be monotonic with regard to probabilities.  
        Namely, the class probability is a monotonic transformation of the Logit.  
        For instance, by a SoftMax.
      - Classes are in the range {0, 1, ..., numCls - 1}.
    """
    
    vYHat  = np.argmax(mScore, axis = 0) #<! Class prediction
    valAcc = np.mean(vYHat == vY)
    
    return valAcc

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