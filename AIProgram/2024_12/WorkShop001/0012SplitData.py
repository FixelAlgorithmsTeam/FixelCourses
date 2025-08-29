# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Object Detection Workshop
# Downloads the Ball & Referee Dataset.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 02/06/2025 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous
import gdown
import json
import os
from platform import python_version
import random
# import warnings
import shutil

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
TEST_FOLDER_NAME  = 'Test'
TRAIN_FOLDER_NAME = 'Train'


# %% Local Packages


# %% Auxiliary Functions

from AuxFun import ConvertPascalVocYolo, ConvertRectPascalVoc, PlotBox, PlotBBox


# %% Parameters

dClass = {
    'small_vehicle': 0,
    'medium_vehicle': 1,
    'large_vehicle': 2,
    'bus': 3,
    'double_trailer_truck': 4,
    'container': 5,
    'heavy_equipment': 6,
    'pylon': 7,
    'small_aircraft': 8,
    'large_aircraft': 9,
    'small_vessel': 10,
    'medium_vessel': 11,
    'large_vessel': 12,
}
lClass = [key for key in dClass.keys()]

dataFolderPath       = os.path.join(DATA_FOLDER_NAME)
yoloImagesFolderPath = os.path.join(dataFolderPath, 'images')
yoloLabelsFolderPath = os.path.join(dataFolderPath, 'labels')


# %% [markdown]
#
# ## Generate / Load Data
# 
# This section is used to download the data set from Google Drive and extract it.      
# The dataset, composed of 300 images, will be extracted to `Data` folder.    
# Once the data is downloaded, it should be labeled using LabelMe or similar tool.    


# %% Load / Generate Data

lImgFiles = os.listdir(yoloImagesFolderPath)
lImgFiles = [imgFile for imgFile in lImgFiles if imgFile.endswith('png')]

lLblFiles = os.listdir(yoloLabelsFolderPath)
lLblFiles = [imgFile for imgFile in lLblFiles if imgFile.endswith('txt')]

lImgFilesBaseName = [os.path.splitext(fileName)[0] for fileName in lImgFiles]
lLblFilesBaseName = [os.path.splitext(fileName)[0] for fileName in lLblFiles]

set(lImgFilesBaseName) == set(lLblFilesBaseName)

lFilesBaseName = lImgFilesBaseName.copy()


# %% Display Image

imgIdx = random.randint(0, len(lFilesBaseName) - 1)

imgFilePath = os.path.join(yoloImagesFolderPath, lFilesBaseName[imgIdx] + '.png')
mI = ski.io.imread(imgFilePath)
mI = ski.util.img_as_float64(mI)

lblFilePath = os.path.join(yoloLabelsFolderPath, lFilesBaseName[imgIdx] + '.txt')
mData       = np.loadtxt(lblFilePath, dtype = float, ndmin = 2, comments = '#')
vLbl        = mData[:, 0].astype(np.uint64)
mBox        = mData[:, 1:]

lLbl = [lClass[lbl] for lbl in vLbl]

hA = PlotBox(mI, vLbl, mBox, lLabelText = lLbl)
hA.set_title(f'Image: {imgIdx}')
hA.legend();


# %%
