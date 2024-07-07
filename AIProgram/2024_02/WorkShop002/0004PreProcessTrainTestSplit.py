# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io/)
# 
# # Object Detection Workshop
# Data Pre Processing:
#  1. Train Test Split.
#  2. Folders by YOLO v8 Convention.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 06/07/2024 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous
import datetime
import gdown
import json
import math
import os
from platform import python_version
import random
import warnings
import shutil
import yaml


# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
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
TEST_FOLDER_NAME  = 'Test'
TRAIN_FOLDER_NAME = 'Train'
VAL_FOLDER_NAME   = 'Validation'
TILES_FOLDER_NAME = 'Tiles'
IMG_FOLDER_NAME   = 'images' #<! Ultralytics is case sensitive
LBL_FOLDER_NAME   = 'labels' #<! Ultralytics is case sensitive
DRIVE_FOLDER_URL  = 'https://drive.google.com/drive/u/2/folders/1wxKIDN777K8kQ4UhJMu5csSbTVXhG7G9'

D_CLS = {'Ball': 0, 'Referee': 1}
L_CLS = ['Ball', 'Referee']


# %% Local Packages


# %% Auxiliary Functions




# %% Parameters

imgFileExt  = 'png'
yoloFileExt = 'txt'

valSetRatio     = 0.2
trainSetRatio   = 1.0 - valSetRatio


# %% Load / Generate Data

dataFolderPath      = os.path.join(os.getcwd(), DATA_FOLDER_NAME)
tilesFolderPath     = os.path.join(dataFolderPath, TILES_FOLDER_NAME)
imagesFolderPath    = os.path.join(dataFolderPath, IMG_FOLDER_NAME)
labelsFolderPath    = os.path.join(dataFolderPath, LBL_FOLDER_NAME)

# Clean Images folder
if os.path.isdir(imagesFolderPath):
    shutil.rmtree(imagesFolderPath) 

os.makedirs(os.path.join(imagesFolderPath, TRAIN_FOLDER_NAME))
os.makedirs(os.path.join(imagesFolderPath, VAL_FOLDER_NAME))

# Clean Labels folder
if os.path.isdir(labelsFolderPath):
    shutil.rmtree(labelsFolderPath) 

os.makedirs(os.path.join(labelsFolderPath, TRAIN_FOLDER_NAME))
os.makedirs(os.path.join(labelsFolderPath, VAL_FOLDER_NAME))

# List of Files
# Valid file: Ends with `png` and has annotation file (Same name, txt)
lFile = []
for ii, fullFileName in enumerate(os.listdir(tilesFolderPath)):
    fileName, fileExt = os.path.splitext(fullFileName)
    if ((imgFileExt in fileExt) and (os.path.isfile(os.path.join(tilesFolderPath, fileName + '.' + yoloFileExt)))):
        lFile.append(fullFileName)

numFiles = len(lFile)


# %% Split Train and Test

random.seed(seedNum) #<! Reproducible on each cell run
numTestFiles = math.ceil(valSetRatio * numFiles)
lValIdx = random.sample(range(numFiles), numTestFiles)

for ii, imgFileName in enumerate(lFile):
    fileName, fileExt = os.path.splitext(imgFileName)
    if ii in lValIdx:
        # Validation Image
        imgFolderPath = os.path.join(imagesFolderPath, VAL_FOLDER_NAME)
        lblFolderPath = os.path.join(labelsFolderPath, VAL_FOLDER_NAME)
    else:
        # Train Image
        imgFolderPath = os.path.join(imagesFolderPath, TRAIN_FOLDER_NAME)
        lblFolderPath = os.path.join(labelsFolderPath, TRAIN_FOLDER_NAME)
    
    lblFileName = fileName + '.' + yoloFileExt
    
    # https://stackoverflow.com/a/30359308/195787
    shutil.copy(os.path.join(tilesFolderPath, imgFileName), imgFolderPath)
    shutil.copy(os.path.join(tilesFolderPath, lblFileName), lblFolderPath)




# %% Display Results


# %%
