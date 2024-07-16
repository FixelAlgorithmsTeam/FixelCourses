# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io/)
# 
# # Image Segmentation Workshop
# Use the pre trained U-Net model on the The Oxford-IIIT Pet Dataset.
# Weights: https://drive.google.com/file/d/15UZlVEjyINpYAibETZGJDdNRsVkBRvBl
# Parameters: https://drive.google.com/file/d/1uL08rL7IO6vv7_X-f4PjLFeyWpaArQ_m
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
from torchmetrics.segmentation import GeneralizedDiceScore 
from torchmetrics.segmentation import MeanIoU 
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

def ModelToMask( tI: torch.Tensor ) -> np.ndarray:

    tI = torch.squeeze(tI, dim = 0)
    mM = torch.argmax(tI, dim = 0)
    mM = mM.cpu().numpy()

    return mM

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
    hA.imshow(mM, interpolation = 'nearest')
    hA.axis('off')
    hA.set_title('Input Mask')

    if (numImg == 3):
        hA = vHa[2]
        hA.imshow(mP, interpolation = 'nearest')
        hA.axis('off')
        hA.set_title('Predicted Mask')
    
    return hF


# %% Parameters

# Data
dataSetPath = os.path.join(DATA_FOLDER_NAME, DATA_SET_FOLDER)

imgSize = 128

vMean = [0.5, 0.5, 0.5]
vStd  = [0.25, 0.25, 0.25]

lClass = ['Pet', 'Background', 'Border']

modelFileName = 'BestModel_2024_07_11_863.pt' #<! https://drive.google.com/file/d/15UZlVEjyINpYAibETZGJDdNRsVkBRvBl
dataFileName  = 'BestModel_2024_07_11_863.npz' #<! https://drive.google.com/file/d/1uL08rL7IO6vv7_X-f4PjLFeyWpaArQ_m/view?usp=drive_link


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

dsImgSeg    = ImageSegmentationDataset(dataSetPath)
dSplitIdx   = np.load(dataFileName) #<! TrainValSplit.npz of the run
vTrainIdx   = dSplitIdx['vTrainIdx']
vValIdx     = dSplitIdx['vValIdx']
lFilterSize = dSplitIdx['lFilterSize']

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
])
oDataTrnsAnn = TorchVisionTrns.Compose([
    TorchVisionTrns.ToImage(),
    SubtractConst(1),
    TorchVisionTrns.Resize(imgSize, interpolation = InterpolationMode.NEAREST),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.ToDtype(torch.long, scale = False),
    SqueezeTrns(dim = 0),
])


# Update the DS transformer
dsImgSeg.transform          = oDataTrnsImg
dsImgSeg.target_transform   = oDataTrnsAnn


# %% Plot with Transform

numSamples  = len(dsImgSeg)
imgIdx      = random.randrange(numSamples)

mI, mM = dsImgSeg[imgIdx]

hF = PlotMasks(np.transpose(mI.cpu().numpy(), (1, 2, 0)), mM.cpu().numpy())
plt.plot()



# %% Load Model

runDevice = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')) #<! MPS or CUDA device
runDevice = torch.device('cpu')

oModel = BuildUNet(3, len(lClass), lFilterSize)
# The saved model is mapped, by default, to the device it was.
# Loading to a "neutral" device: CPU.
# See: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices.
dModel = torch.load(modelFileName, map_location = 'cpu') #<! Loads saved data
oModel.load_state_dict(dModel['Model'])
oModel = oModel.eval()

torchinfo.summary(oModel, (1, 3, imgSize, imgSize), col_names = ['kernel_size', 'output_size', 'num_params'], device = 'cpu')

oModel = oModel.to(runDevice) #<! We could leave it on CPU as well


# %% Display Results



# %% Display Prediction

# Train
numSamples  = len(vTrainIdx)
imgIdx      = random.randrange(numSamples)
imgIdx      = vTrainIdx[imgIdx]

mI, mM = dsImgSeg[imgIdx]
tI = mI.to(runDevice)
tI = tI[None, :, :, :]
tO = oModel(tI)
mP = ModelToMask(tO)

hF = PlotMasks(np.transpose(mI.cpu().numpy(), (1, 2, 0)), mM.cpu().numpy(), mP = mP)
plt.plot()

# Validation
numSamples  = len(vValIdx)
imgIdx      = random.randrange(numSamples)
imgIdx      = vValIdx[imgIdx]

mI, mM = dsImgSeg[imgIdx]
tI = mI.to(runDevice)
tI = tI[None, :, :, :]
tO = oModel(tI)
mP = ModelToMask(tO)

hF = PlotMasks(np.transpose(mI.cpu().numpy(), (1, 2, 0)), mM.cpu().numpy(), mP = mP)
plt.plot()

# Validation Dice Score (F1 Like) & IoU Score
hDiceScore = GeneralizedDiceScore(num_classes = len(lClass))
hIoUScore  = MeanIoU(num_classes = len(lClass))
vDiceScore = np.zeros(shape = len(vValIdx))
vIouScore  = np.zeros(shape = len(vValIdx))
for ii, imgIdx in enumerate(vValIdx):
    mI, mM = dsImgSeg[imgIdx]
    tI = mI.to(runDevice)
    tI = tI[None, :, :, :]
    tO = oModel(tI)
    tO = tO.to('cpu').detach()
    
    if imgIdx in [6010, 4710, 1105]: #<! Fails
        continue

    vDiceScore[ii] = hDiceScore(torch.argmax(tO, dim = 1), mM[None, :, :]).item()
    vIouScore[ii]  = hIoUScore(torch.argmax(tO, dim = 1), mM[None, :, :]).item()

# Display Score per Image
hF, hA = plt.subplots(figsize = (12, 5))

hA.plot(range(len(vValIdx)), vDiceScore, label = 'Dice Score')
hA.plot(range(len(vValIdx)), vIouScore, label = 'Mean IoU Score')
hA.set_title('Dice and Mean IoU Score')
hA.set_xlabel('Image Index')
hA.set_ylabel('Score')

hA.legend();

# !!Tasks:
# - Show the performance per class.
# - Display confusion matrix.


# %%
