
# Python STD
from enum import auto, Enum, unique
import math

# Data
import numpy as np
# import pandas as pd
import scipy as sp

# Machine Learning

# Deep Learning
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous
import time

# Course Packages
from DeepLearningBlocks import NNMode


# Typing
from typing import Callable, Dict, Generator, List, Optional, Set, Tuple, Union


# Auxiliary Functions

def ResetModelWeights( oModel: nn.Module ) -> None:
    # https://discuss.pytorch.org/t/19180

    for layer in oModel.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
            for child in layer.children():
                ResetModelWeights(child)

# Initialization Function

def InitWeights( oLayer: nn.Module ) -> None:
    # Use: oModel.apply(InitWeights) #<! Applies the function on all layers
    if isinstance(oLayer, nn.Linear):
        nn.init.kaiming_normal_(oLayer.weight.data)


def RunEpoch( oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, opMode: NNMode = NNMode.TRAIN ) -> Tuple[float, float]:
    """
    Runs a single Epoch (Train / Test) of a model.  
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlData      - PyTorch `Dataloader` object.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oOpt        - PyTorch `Optimizer` object.
        opMode      - An `NNMode` to set the mode of operation.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer is required for training mode.
    """
    
    epochLoss   = 0.0
    epochScore  = 0.0
    numSamples  = 0
    numBatches = len(dlData)

    runDevice = next(oModel.parameters()).device #<! CPU \ GPU

    if opMode == NNMode.TRAIN:
        oModel.train(True) #<! Equivalent of `oModel.train()`
    elif opMode == NNMode.INFERENCE:
        oModel.eval() #<! Equivalent of `oModel.train(False)`
    else:
        raise ValueError(f'The `opMode` value {opMode} is not supported!')
    
    for ii, (mX, vY) in enumerate(dlData):
        # Move Data to Model's device
        mX = mX.to(runDevice) #<! Lazy
        vY = vY.to(runDevice) #<! Lazy


        batchSize = mX.shape[0]
        
        if opMode == NNMode.TRAIN:
            # Forward
            mZ      = oModel(mX) #<! Model output
            valLoss = hL(mZ, vY) #<! Loss
            
            # Backward
            oOpt.zero_grad()    #<! Set gradients to zeros
            valLoss.backward()  #<! Backward
            oOpt.step()         #<! Update parameters
        else: #<! Value of `opMode` was already validated
            with torch.no_grad():
                # No computational 
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss

        with torch.no_grad():
            # Score
            valScore = hS(mZ, vY)
            # Normalize so each sample has the same weight
            epochLoss  += batchSize * valLoss.item()
            epochScore += batchSize * valScore.item()
            numSamples += batchSize

        print(f'\r{"Train" if opMode == NNMode.TRAIN else "Val"} - Iteration: {ii:3d} ({numBatches}): loss = {valLoss:.6f}', end = '')
    
    print('', end = '\r')
            
    return epochLoss / numSamples, epochScore / numSamples

# Training Model Loop Function

def TrainModel( oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, numEpoch: int, hL: Callable, hS: Callable , oTBWriter: Optional[SummaryWriter] = None) -> Tuple[nn.Module, List, List, List, List]:

    lTrainLoss  = []
    lTrainScore = []
    lValLoss    = []
    lValScore   = []

    # Support R2
    bestScore = -1e9 #<! Assuming higher is better

    for ii in range(numEpoch):
        startTime           = time.time()
        trainLoss, trainScr = RunEpoch(oModel, dlTrain, hL, hS, oOpt, opMode = NNMode.TRAIN) #<! Train
        valLoss,   valScr   = RunEpoch(oModel, dlVal, hL, hS, oOpt, opMode = NNMode.INFERENCE)    #<! Score Validation
        epochTime           = time.time() - startTime

        # Aggregate Results
        lTrainLoss.append(trainLoss)
        lTrainScore.append(trainScr)
        lValLoss.append(valLoss)
        lValScore.append(valScr)

        if oTBWriter is not None:
            oTBWriter.add_scalar('Train Loss', trainLoss, ii)
            oTBWriter.add_scalar('Train Score', trainScr, ii)
            oTBWriter.add_scalar('Validation Loss', valLoss, ii)
            oTBWriter.add_scalar('Validation Score', valScr, ii)
        
        # Display (Babysitting)
        print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}:', end = '')
        print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
        print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
        print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
        print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
        print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

        # Save best model ("Early Stopping")
        if valScr > bestScore:
            bestScore = valScr
            print(' | <-- Checkpoint!', end = '')
            try:
                dCheckpoint = {'Model' : oModel.state_dict(), 'Optimizer' : oOpt.state_dict()}
                torch.save(dCheckpoint, 'BestModel.pt')
            except:
                pass
        print(' |')
    
    # Load best model ("Early Stopping")
    dCheckpoint = torch.load('BestModel.pt')
    oModel.load_state_dict(dCheckpoint['Model'])

    return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore
