# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io/)
# 
# # Image Segmentation Workshop
# Trains a U-Net model on the The Oxford-IIIT Pet Dataset.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 10/07/2024 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchinfo
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as TorchVisionTrns


# Miscellaneous
import datetime
from enum import auto, Enum, unique
import os
from platform import python_version
import random
import time
import warnings


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# Matplotlib default color palette
lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# sns.set_theme() #>! Apply SeaBorn theme


# %% Constants

FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2

DATA_FOLDER_NAME  = 'Data'
DATA_SET_FOLDER   = 'OxfordIIITPet'


# %% Local Packages

from DataLoader import ImageSegmentationDataset
from UNetModule import BuildUNet


# %% Auxiliary Functions

@unique
class NNMode(Enum):
    TRAIN     = auto()
    INFERENCE = auto() 


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

def ModelToImg( tI: torch.Tensor ) -> np.ndarray:

    tI = torch.squeeze(tI, dim = 0)
    mM = torch.argmin(tI, dim = 0)
    mM = mM.cpu().numpy()

    return np.transpose(mM, (1, 2, 0))

def PlotMasks( mI: np.ndarray, mM: np.ndarray, *, mP: Optional[np.ndarray] = None ) -> plt.Figure:

    if mP is not None:
        numImg = 3
    else:
        numImg = 2
    
    hF, vHa = plt.subplots(nrows = 1, ncols = numImg, figsize = (5 * numImg, 5))

    vHa = vHa.flat
    hA = vHa[0]
    hA.imshow(mI)
    hA.axis('off')
    hA.set_title('Input Image')

    hA = vHa[1]
    hA.imshow(mM)
    hA.axis('off')
    hA.set_title('Input Mask')

    if (numImg == 3):
        hA = vHa[2]
        hA.imshow(mP)
        hA.axis('off')
        hA.set_title('Predicted Mask')
    
    return hF


# %% Parameters

# Data
dataSetPath = os.path.join(DATA_FOLDER_NAME, DATA_SET_FOLDER)

imgSize = 128

vMean = [0.5, 0.5, 0.5]
vStd  = [0.25, 0.25, 0.25]

numSamplsTrain = 6000
numSamplesVal = 1390

lClass = ['Pet', 'Background', 'Border']

# Model
lFilterSize = [10, 20, 40] #<! Assumption: filter_size[ii + 1] == 2 * filter_size[ii]

# Training
batchSize   = 200
numWork     = 2 #<! Number of workers
nEpochs     = 45

# %% [markdown]

## The Oxford-IIIT Pet Dataset
# 
# The [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) is built from 7349 images of dogs and cats.
# 
# ![](https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg)
# 
# Each image is annotated with a mask with the labels:
#  * `1` - Pet pixel.
#  * `2` - Background pixel.
#  * `3` - Border pixel.
#
# The labels will be mapped into `[0, 1, 2]`.


# %% Load / Generate Data

dsImgSeg = ImageSegmentationDataset(dataSetPath)

# Plot Samples

numSamples  = len(dsImgSeg)
imgIdx      = random.randrange(numSamples)

mI, mM = dsImgSeg[imgIdx]

hF = PlotMasks(mI, mM)
plt.plot()


# %% Data Transforms

oDataTrnsImg = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    TorchVisionTrns.ToDtype(torch.float32, scale = True),
    TorchVisionTrns.Resize(imgSize),
    TorchVisionTrns.CenterCrop(imgSize),
    # TorchVisionTrns.Normalize(mean = vMean, std = vStd),
])
oDataTrnsAnn = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    TorchVisionTrns.Lambda(lambda x: x - 1),
    TorchVisionTrns.Resize(imgSize, interpolation = InterpolationMode.NEAREST_EXACT),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.ToDtype(torch.long, scale = False),
    TorchVisionTrns.Lambda(lambda x: torch.squeeze(x, dim = 0)),
])


# Update the DS transformer
dsImgSeg.transform          = oDataTrnsImg
dsImgSeg.target_transform   = oDataTrnsAnn


# %% Train Test Split

dsTrain, dsVal = torch.utils.data.random_split(dsImgSeg, (numSamplsTrain, numSamplesVal))


# %% Data Loaders

# dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWork, persistent_workers = True)
# dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWork, persistent_workers = True)

dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = 0, persistent_workers = False)
dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = 0, persistent_workers = False)

# Iterate on the Loader
# The first batch.
tX, vY = next(iter(dlTrain)) #<! PyTorch Tensors

print(f'The batch features dimensions: {tX.shape}')
print(f'The batch labels dimensions: {vY.shape}')

# Looping
for ii, (tX, vY) in zip(range(1), dlVal): #<! https://stackoverflow.com/questions/36106712
    print(f'The batch features dimensions: {tX.shape}')
    print(f'The batch labels dimensions: {vY.shape}')


# %% Build Model

oModel = BuildUNet(3, 3, lFilterSize)

torchinfo.summary(oModel, (batchSize, 3, imgSize, imgSize), col_names = ['kernel_size', 'output_size', 'num_params'], device = 'cpu')


# %% Train Model

# Run Device

runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #<! The 1st CUDA device
oModel    = oModel.to(runDevice) #<! Transfer model to device

# Loss and Score Function

hL = nn.CrossEntropyLoss()
hS = MulticlassAccuracy(num_classes = len(lClass), average = 'micro')
hL = hL.to(runDevice) #<! Not required!
hS = hS.to(runDevice)

# Define Optimizer

oOpt = torch.optim.AdamW(oModel.parameters(), lr = 1e-3, betas = (0.9, 0.99), weight_decay = 1e-3) #<! Define optimizer

# Define Scheduler

oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 5e-3, total_steps = nEpochs)

# Train Model

oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, nEpochs, hL, hS, oSch = oSch)



# %% Display Results

# Plot Training Phase

hF, vHa = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 5))
vHa = np.ravel(vHa)

hA = vHa[0]
hA.plot(lTrainLoss, lw = 2, label = 'Train')
hA.plot(lValLoss, lw = 2, label = 'Validation')
hA.set_title('Cross Entropy Loss')
hA.set_xlabel('Epoch')
hA.set_ylabel('Loss')
hA.legend()

hA = vHa[1]
hA.plot(lTrainScore, lw = 2, label = 'Train')
hA.plot(lValScore, lw = 2, label = 'Validation')
hA.set_title('Accuracy Score')
hA.set_xlabel('Epoch')
hA.set_ylabel('Score')
hA.legend()

hA = vHa[2]
hA.plot(lLearnRate, lw = 2)
hA.set_title('Learn Rate Scheduler')
hA.set_xlabel('Epoch')
hA.set_ylabel('Learn Rate')

