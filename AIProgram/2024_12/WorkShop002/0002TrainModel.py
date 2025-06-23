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
import torchinfo
from torchmetrics.classification import MulticlassAccuracy
from torchvision import tv_tensors
from torchvision.transforms import v2 as TorchVisionTrns


# Miscellaneous
import os
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

from DL import BuildUNet, ImageSegmentationDataset
from DL import GenDataLoaders, TrainModel
from AuxFun import DataTensorToImageMask, ModelToMask, PlotMasks


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
lFilterSize = [10, 20, 40] #<! Assumption: filter_size[ii + 1] == 2 * filter_size[ii]

# Training
batchSize   = 256
numWork     = 2 #<! Number of workers
nEpochs     = 5 #<! Use the script `0002TrainModelScript.py` to run for more epochs

# %% [markdown]

## The Oxford-IIIT Pet Dataset
# 
# The [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) is built from 7349 images of dogs and cats.
# 
# ![](https://www.robots.ox.ac.uk/~vgg/data/pets/pet_annotations.jpg) <!-- https://i.imgur.com/Dgg8XwM.jpeg -->
# 
# Each image is annotated with a mask with the labels:
#  * `1` - Pet pixel.
#  * `2` - Background pixel.
#  * `3` - Border pixel.
#
# The labels will be mapped into `[0, 1, 2]`.


# %% Load / Generate Data

hDataTensorToImageMask = lambda tI, tM: DataTensorToImageMask(tI, tM, lMean, lStd)
dsImgSeg = ImageSegmentationDataset(dataSetPath)

# Plot Samples

numSamples  = len(dsImgSeg)
imgIdx      = random.randrange(numSamples)

tI, tM = dsImgSeg[imgIdx]

mI = np.transpose(tI.cpu().numpy(), (1, 2, 0)) #<! (C x H x W) -> (H x W x C)
mM = tM.cpu().numpy() #<! (H x W)

hF = PlotMasks(mI, mM)
# plt.plot()


# %% Data Transforms

# Using TorchVision Transform v2
# The v2 Resize automatically sets the Mask interpolation to _Nearest Neighbor_.
oDataTrns = TorchVisionTrns.Compose([
    TorchVisionTrns.Resize(imgSize),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.RandomHorizontalFlip(p = 0.25),
    TorchVisionTrns.RandomGrayscale(p = 0.1),
    TorchVisionTrns.ToDtype(dtype = {tv_tensors.Image: torch.float32, 'others': None}, scale = True),
    TorchVisionTrns.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.05),
    TorchVisionTrns.Normalize(mean = lMean, std = lStd),
    TorchVisionTrns.ToDtype(dtype = {tv_tensors.Mask: torch.int64, 'others': None}, scale = False),
])

# Update the DS transformer
dsImgSeg.transform = oDataTrns


# %% Train Test Split

oRng = torch.Generator().manual_seed(seedNum)
dsTrain, dsVal = torch.utils.data.random_split(dsImgSeg, (numSamplsTrain, numSamplesVal), oRng)

numSamples  = len(dsTrain)
imgIdx      = random.randrange(numSamples)

mI, mM = dsTrain[imgIdx]
mI, mM = hDataTensorToImageMask(mI, mM)

hF = PlotMasks(mI, mM)
plt.plot()


# %% Data Loaders

# On Windows, numWorkers = 0 is required in Jupyter mode.
# Hence to actually run, use the script `0002TrainModelScript`.
dlTrain, dlVal = GenDataLoaders(dsTrain, dsVal, batchSize, numWorkers = 0, dropLast = True, PersWork = False)

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

oModel = BuildUNet(3, len(lClass), lFilterSize)

torchinfo.summary(oModel, (batchSize, 3, imgSize, imgSize), col_names = ['kernel_size', 'output_size', 'num_params'], device = 'cpu')


# %% Train Model

# Run Device

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
hA.set_ylabel('Learn Rate');


# %% Display Prediction

oModel = BuildUNet(3, len(lClass), lFilterSize)
dModel = torch.load('BestModel.pt') #<! Loads saved data
oModel.load_state_dict(dModel['Model'])
oModel = oModel.eval()
oModel = oModel.to(runDevice)

numSamples  = len(dsVal)
imgIdx      = random.randrange(numSamples)

tI, tM = dsVal[imgIdx]
tI = tI.to(runDevice)
tI = tI[None, :, :, :]
with torch.inference_mode():
    tO = oModel(tI)
mP = ModelToMask(tO)

tI = tI.to('cpu')
mI, mM = hDataTensorToImageMask(tI, tM)

hF = PlotMasks(mI, mM, mP = mP)


# %%
