# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
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
torch.manual_seed(seedNum)

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

from DL import AdjustMask, BuildUNet, ImageSegmentationDataset
from DL import GenDataLoaders, RunEpoch, TrainModel


# %% Auxiliary Functions

class SqueezeTrns(nn.Module):
    def __init__(self, dim: int = None) -> None:
        super().__init__()

        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.squeeze(x, dim = self.dim)

class SubtractConst(nn.Module):
    def __init__(self, const: int = 0) -> None:
        super().__init__()

        self.const = const
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x - self.const


# %% Parameters

# Data
dataSetPath = os.path.join(DATA_FOLDER_NAME, DATA_SET_FOLDER)

imgSize = 128

vMean = [0.5, 0.5, 0.5]
vStd  = [0.25, 0.25, 0.25]

numSamplsTrain = 6000
numSamplesVal  = 1390

lClass = ['Pet', 'Background', 'Border']

# Model
lFilterSize = [10, 20, 40, 80] #<! Assumption: filter_size[ii + 1] == 2 * filter_size[ii]

# Training
batchSize   = 250
numWorkers  = 4 #<! Number of workers
nEpochs     = 100


# %% Load / Generate Data

dsImgSeg = ImageSegmentationDataset(dataSetPath)


# %% Data Transforms

oDataTrnsImg = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    TorchVisionTrns.ToDtype(torch.float32, scale = True),
    TorchVisionTrns.Resize(imgSize),
    TorchVisionTrns.CenterCrop(imgSize),
    # TorchVisionTrns.Normalize(mean = vMean, std = vStd),
])
# Lambda functions prevent Multi Threading on Windows
oDataTrnsAnn = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    # TorchVisionTrns.Lambda(lambda x: x - 1), #<! Cause issues with Multi Processing
    SubtractConst(1),
    TorchVisionTrns.Resize(imgSize, interpolation = InterpolationMode.NEAREST),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.ToDtype(torch.long, scale = False),
    # TorchVisionTrns.Lambda(lambda x: torch.squeeze(x, dim = 0)), #<! Cause issues with Multi Processing
    SqueezeTrns(dim = 0),
])
# oDataTrnsAnn = TorchVisionTrns.Compose([
#     TorchVisionTrns.ToImage(),
#     TorchVisionTrns.Resize(imgSize, interpolation = InterpolationMode.NEAREST),
#     TorchVisionTrns.CenterCrop(imgSize),
#     TorchVisionTrns.ToDtype(torch.long, scale = False),
#     AdjustMask()
# ])


# Update the DS transformer
dsImgSeg.transform          = oDataTrnsImg
dsImgSeg.target_transform   = oDataTrnsAnn


# %% Train Test Split

oRng = torch.Generator().manual_seed(seedNum)
dsTrain, dsVal = torch.utils.data.random_split(dsImgSeg, (numSamplsTrain, numSamplesVal), oRng)

vTrainIdx   = np.array(dsTrain.indices)
vValIdx     = np.array(dsVal.indices)
np.savez('TrainValSplit', vTrainIdx = vTrainIdx, vValIdx = vValIdx, lFilterSize = lFilterSize)


# %% Data Loaders


def Main(dsTrain, dsVal, batchSize, numWorkers, lClass, lFilterSize, nEpochs):
    
    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWorkers, persistent_workers = True)
    dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWorkers, persistent_workers = True)
    
    oModel = BuildUNet(3, len(lClass), lFilterSize)
    
    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) #<! The 1st CUDA device
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



# %% Main

if __name__ == '__main__':
    Main(dsTrain, dsVal, batchSize, numWorkers, lClass, lFilterSize, nEpochs)
