# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Image Segmentation Workshop
# Use the pre trained U-Net model on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets).
# - Weights: https://drive.google.com/file/d/15UZlVEjyINpYAibETZGJDdNRsVkBRvBl.
# - Parameters: https://drive.google.com/file/d/1uL08rL7IO6vv7_X-f4PjLFeyWpaArQ_m.
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
from torchmetrics.segmentation import GeneralizedDiceScore 
from torchmetrics.segmentation import MeanIoU 
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

from DL import BuildUNet, ImageSegmentationDataset
from AuxFun import DataTensorToImageMask, UnNormalizeImg, ModelToMask, PlotMasks


# %% Auxiliary Functions


# %% Parameters

# Data
dataSetPath = os.path.join(DATA_FOLDER_NAME, DATA_SET_FOLDER)

imgSize = 128

lMean = [0.5, 0.5, 0.5]
lStd  = [0.25, 0.25, 0.25]

lClass = ['Pet', 'Background', 'Border']

modelFileName = 'BestModel_2025_06_23_875.pt' #<! https://drive.google.com/file/d/1mPqvi9Mhvhz5GdtLtRtUDl2jVRbzzqRx
dataFileName  = 'BestModel_2025_06_23_875_Train_Test_Split.npz' #<! https://drive.google.com/file/d/17zc8fLU_vUPns8Vb-Xg_PAgnqQs6WDac
trainFileName = 'BestModel_2025_06_23_875_Training_Data.pkl' #<! https://drive.google.com/file/d/1woaRDepTUY6uaOebx3B0ste6nYuC7JJB


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

hDataTensorToImageMask = lambda tI, tM: DataTensorToImageMask(tI, tM, lMean, lStd)

dsImgSeg    = ImageSegmentationDataset(dataSetPath)
dSplitIdx   = np.load(dataFileName) #<! TrainValSplit.npz of the run
vTrainIdx   = dSplitIdx['vTrainIdx']
vValIdx     = dSplitIdx['vValIdx']
lFilterSize = dSplitIdx['lFilterSize']

# Plot Samples

numSamples  = len(dsImgSeg)
imgIdx      = random.randrange(numSamples)

tI, tM = dsImgSeg[imgIdx]

mI = np.permute_dims(tI.cpu().numpy(), (1, 2, 0)) #<! (C, H, W) -> (H, W, C)
mM = tM.cpu().numpy()

hF = PlotMasks(mI, mM)


# %% Plot Training Phase

# Load Training Phase Data
try:
    with open(trainFileName, 'rb') as file:
        dTrainPhase = pickle.load(file)
except FileNotFoundError:
    print(f"Error: The file '{trainFileName}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

lTrainLoss  = dTrainPhase['lTrainLoss']
lTrainScore = dTrainPhase['lTrainScore']
lValLoss    = dTrainPhase['lValLoss']
lValScore   = dTrainPhase['lValScore']
lLearnRate  = dTrainPhase['lLearnRate']

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


# %% Data Transforms

# Using TorchVision Transform v2
# The v2 Resize automatically sets the Mask interpolation to _Nearest Neighbor_.
oDataTrns = TorchVisionTrns.Compose([
    TorchVisionTrns.Resize(imgSize),
    TorchVisionTrns.CenterCrop(imgSize),
    TorchVisionTrns.ToDtype(dtype = {tv_tensors.Image: torch.float32, 'others': None}, scale = True),
    TorchVisionTrns.Normalize(mean = lMean, std = lStd),
])

# Update the DS transformer
dsImgSeg.transform = oDataTrns


# %% Plot with Transform

numSamples  = len(dsImgSeg)
imgIdx      = random.randrange(numSamples)

tI, tM = dsImgSeg[imgIdx]

# Compensate for the normalization
mI, mM = hDataTensorToImageMask(tI, tM)
# tI = UnNormalizeImg(tI, lMean, lStd)
# mI = np.permute_dims(tI.cpu().numpy(), (1, 2, 0)) #<! (C, H, W) -> (H, W, C)
# mM = tM.cpu().numpy()

hF = PlotMasks(mI, mM)


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

# %% Model Summary

torchinfo.summary(oModel, (1, 3, imgSize, imgSize), col_names = ['kernel_size', 'output_size', 'num_params'], device = 'cpu')


# %% Move Model to Device

oModel = oModel.to(runDevice) #<! We could leave it on CPU as well


# %% Display Results


# %% Display Prediction

# Train Dataset
numSamples  = len(vTrainIdx)
imgIdx      = random.randrange(numSamples)
imgIdx      = vTrainIdx[imgIdx]

mI, tM = dsImgSeg[imgIdx]
tI = mI.to(runDevice)
tI = tI[None, :, :, :]
with torch.inference_mode():
    tO = oModel(tI)
mP = ModelToMask(tO)

mI, mM = hDataTensorToImageMask(tI, tM)

# tI = torch.squeeze(tI, dim = 0) #<! Remove batch dimension

# tI = UnNormalizeImg(tI, lMean, lStd)
# mI = np.permute_dims(tI.cpu().numpy(), (1, 2, 0)) #<! (C, H, W) -> (H, W, C)
# mM = tM.cpu().numpy()

hF = PlotMasks(mI, mM, mP = mP)

# Validation Dataset
numSamples  = len(vValIdx)
imgIdx      = random.randrange(numSamples)
imgIdx      = vValIdx[imgIdx]

mI, tM = dsImgSeg[imgIdx]
tI = mI.to(runDevice)
tI = tI[None, :, :, :]
with torch.inference_mode():
    tO = oModel(tI)
mP = ModelToMask(tO)

mI, mM = hDataTensorToImageMask(tI, tM)

# tI = torch.squeeze(tI, dim = 0) #<! Remove batch dimension

# tI = UnNormalizeImg(tI, lMean, lStd)
# mI = np.permute_dims(tI.cpu().numpy(), (1, 2, 0)) #<! (C, H, W) -> (H, W, C)
# mM = tM.cpu().numpy()

hF = PlotMasks(mI, mM, mP = mP)

# Validation Dice Score (F1 Like) & IoU Score
hDiceScore = GeneralizedDiceScore(num_classes = len(lClass))
hIoUScore  = MeanIoU(num_classes = len(lClass))
vDiceScore = np.zeros(shape = len(vValIdx))
vIouScore  = np.zeros(shape = len(vValIdx))
for ii, imgIdx in enumerate(vValIdx):
    mI, mM = dsImgSeg[imgIdx]
    tI = mI.to(runDevice)
    tI = tI[None, :, :, :]
    with torch.inference_mode():
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
