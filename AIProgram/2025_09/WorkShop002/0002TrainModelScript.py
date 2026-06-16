# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Image Segmentation Workshop
# Trains a U-Net model on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets).
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.1.001 | 07/07/2025 | Royi Avital | Reordered the augmentation (Color Jitter and Grayscale)                                  |
# | 1.1.000 | 10/06/2025 | Royi Avital | Using TorchVision Transform v2                                                           |
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
from torchmetrics.classification import MulticlassAccuracy
from torchvision import tv_tensors
from torchvision.transforms import v2 as TorchVisionTrns

# Miscellaneous
import os
import pickle
from platform import python_version
import random
# import warnings


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

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

from DL import BuildUNet, ImageSegmentationDataset
from DL import TrainModel


# %% Auxiliary Functions


# %% Parameters

# Data
dataSetPath = os.path.join(DATA_FOLDER_NAME, DATA_SET_FOLDER)

imgSize = 128

lMean = [0.5, 0.5, 0.5]
lStd  = [0.25, 0.25, 0.25]

numSamplsTrain = 6000
numSamplesVal  = 1390

lClass = ['Pet', 'Background', 'Border']

# Model
lFilterSize = [10, 20, 40, 80] #<! Assumption: filter_size[ii + 1] == 2 * filter_size[ii]

# Training
batchSize   = 512
numWorkers  = 8 #<! Number of workers
nEpochs     = 125


# %% Load / Generate Data

dsImgSeg = ImageSegmentationDataset(dataSetPath)


# %% Data Transforms

# Using TorchVision Transform v2
# The v2 Resize automatically sets the Mask interpolation to _Nearest Neighbor_.
oDataTrns = TorchVisionTrns.Compose([
    TorchVisionTrns.Resize(imgSize),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.RandomHorizontalFlip(p = 0.25),
    TorchVisionTrns.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.05),
    TorchVisionTrns.RandomGrayscale(p = 0.1),
    TorchVisionTrns.ToDtype(dtype = {tv_tensors.Image: torch.float32, 'others': None}, scale = True),
    TorchVisionTrns.Normalize(mean = lMean, std = lStd),
    TorchVisionTrns.ToDtype(dtype = {tv_tensors.Mask: torch.int64, 'others': None}, scale = False),
])

# Update the DS transformer
dsImgSeg.transform = oDataTrns


# %% Train Test Split

oRng = torch.Generator().manual_seed(seedNum)
dsTrain, dsVal = torch.utils.data.random_split(dsImgSeg, (numSamplsTrain, numSamplesVal), oRng)

vTrainIdx   = np.array(dsTrain.indices)
vValIdx     = np.array(dsVal.indices)
np.savez('TrainValSplit', vTrainIdx = vTrainIdx, vValIdx = vValIdx, lFilterSize = lFilterSize)


# %% Main Function

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

    # Save the training results using pickle
    with open('TrainResults.pkl', 'wb') as hF:
        pickle.dump({
            'lTrainLoss': lTrainLoss,
            'lTrainScore': lTrainScore,
            'lValLoss': lValLoss,
            'lValScore': lValScore,
            'lLearnRate': lLearnRate
        }, hF)


# %% Main

if __name__ == '__main__':
    Main(dsTrain, dsVal, batchSize, numWorkers, lClass, lFilterSize, nEpochs)

# %% [markdown]
# * <font color='blue'>(**!**)</font> Use _Weights and Biases_ to tune the _Hyper Parameters_ of the model.

