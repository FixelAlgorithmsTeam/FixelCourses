
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
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

# Image Processing / Computer Vision

# Optimization

# Python STD
import os

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous
import time

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Auxiliary Classes

@unique
class NNMode(Enum):
    TRAIN     = auto()
    INFERENCE = auto() 

class TBLogger():
    def __init__( self, logDir: Optional[str] = None ) -> None:

        self.oTBWriter  = SummaryWriter(log_dir = logDir)
        self.iiEpcoh    = 0
        self.iiItr      = 0
        
        pass

    def close( self ) -> None:

        self.oTBWriter.close()


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
                # No computational graph
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss

        with torch.no_grad():
            # Score
            valScore = hS(mZ, vY)
            # Normalize so each sample has the same weight
            epochLoss  += batchSize * valLoss.item()
            epochScore += batchSize * valScore.item()
            numSamples += batchSize

        print(f'\r{"Train" if opMode == NNMode.TRAIN else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}, loss: {valLoss:.6f}', end = '')
    
    print('', end = '\r')
            
    return epochLoss / numSamples, epochScore / numSamples


# Training Model Loop Function

def TrainModel( oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, numEpoch: int, hL: Callable, hS: Callable, *, oSch: Optional[LRScheduler] = None, oTBWriter: Optional[SummaryWriter] = None) -> Tuple[nn.Module, List, List, List, List]:
    """
    Trains a model given test and validation data loaders.  
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlTrain     - PyTorch `Dataloader` object (Training).
        dlVal       - PyTorch `Dataloader` object (Validation).
        oOpt        - PyTorch `Optimizer` object.
        numEpoch    - Number of epochs to run.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oSch        - PyTorch `Scheduler` (`LRScheduler`) object.
        oTBWriter   - PyTorch `SummaryWriter` object (TensorBoard).
    Output:
        lTrainLoss     - Scalar of the loss.
        lTrainScore    - Scalar of the score.
        lValLoss    - Scalar of the score.
        lValScore    - Scalar of the score.
        lLearnRate    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer is required for training mode.
    """

    lTrainLoss  = []
    lTrainScore = []
    lValLoss    = []
    lValScore   = []
    lLearnRate  = []

    # Support R2
    bestScore = -1e9 #<! Assuming higher is better

    learnRate = oOpt.param_groups[0]['lr']

    for ii in range(numEpoch):
        startTime           = time.time()
        trainLoss, trainScr = RunEpoch(oModel, dlTrain, hL, hS, oOpt, opMode = NNMode.TRAIN) #<! Train
        valLoss,   valScr   = RunEpoch(oModel, dlVal, hL, hS, oOpt, opMode = NNMode.INFERENCE) #<! Score Validation
        if oSch is not None:
            # Adjusting the scheduler on Epoch level
            learnRate = oSch.get_last_lr()[0]
            oSch.step()
        epochTime           = time.time() - startTime

        # Aggregate Results
        lTrainLoss.append(trainLoss)
        lTrainScore.append(trainScr)
        lValLoss.append(valLoss)
        lValScore.append(valScr)
        lLearnRate.append(learnRate)

        if oTBWriter is not None:
            oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
            oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
            oTBWriter.add_scalar('Learning Rate', learnRate, ii)
        
        # Display (Babysitting)
        print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}', end = '')
        print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
        print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
        print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
        print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
        print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

        # Save best model ("Early Stopping")
        if valScr > bestScore:
            bestScore = valScr
            try:
                dCheckPoint = {'Model': oModel.state_dict(), 'Optimizer': oOpt.state_dict()}
                if oSch is not None:
                    dCheckPoint['Scheduler'] = oSch.state_dict()
                torch.save(dCheckPoint, 'BestModel.pt')
                print(' | <-- Checkpoint!', end = '')
            except:
                print(' | <-- Failed!', end = '')
        print(' |')
    
    # Load best model ("Early Stopping")
    # dCheckPoint = torch.load('BestModel.pt')
    # oModel.load_state_dict(dCheckPoint['Model'])

    return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate
