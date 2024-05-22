# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io/)
# 
# # Fixel Algorithms - Generate Not MNIST
# Generates the 
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks
# - Requires installing PyAV: `micromamba install av -c conda-forge`.
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 19/02/2024 | Royi Avital | First version                                                                            |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Optimization

# Miscellaneous
# import glob
# import datetime
import os
import pickle
from platform import python_version
import random
# import warnings
# import yaml

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns



# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_NAME        = 'ValidIt'
DATA_FOLDER         = 'Data'
IMG_FOLDER          = 'Images'
MISC_FOLDER         = 'Miscellaneous'
MODELS_FOLDER       = 'Models'
TIME_SERIES_FOLDER  = 'TimeSeries'
VIDEO_FOLDER        = 'Videos'

PROJECT_BASE_FOLDER = os.getcwd()[:(os.getcwd().lower().find(PROJECT_NAME.lower()) + len(PROJECT_NAME))]

FREQ_TO_RR_FACTOR = 60
FREQ_TO_HR_FACTOR = 60


# %% Local Packages



# %% Auxiliary Functions




# %% Parameters

dirName = 'NotMNIST'
fileName = 'NotMNIST.mat'


# %% Generate / Load Data

dirPath = os.path.join('.', dirName)
lDir = [os.path.join(dirPath, dirName) for dirName in os.listdir(dirPath) if os.path.isdir(os.path.join(dirPath, dirName))]


# %% Build Data 

lX = []
lY = []

for ii, imgDirPath in enumerate(lDir):
    lImgFiles = os.listdir(imgDirPath)
    for jj, imgFileName in enumerate(lImgFiles):
        # dirPath = os.path.dirname() #<! Dir Path
        # fileName = os.path.basename() #<! Full File Name
        _, fileExt = os.path.splitext(imgFileName)
        filePath = os.path.join(imgDirPath, imgFileName)
        if ((fileExt == '.png') and (os.path.getsize(filePath) > 1)):
            # Some files are dead (Size ~ 0)
            mI = ski.io.imread(filePath)
            lX.append(np.ravel(mI))
            lY.append(ii)


# %% Save as MAT (MATLAB)

vY = np.array(lY)
mX = np._r[lX]

sp.io.savemat('NotMNIST.mat', {'mX': mX, 'vY': vY})


