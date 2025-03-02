# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Auxiliary Scripts - Filter YOLO Dataset
# Filters YOLO structured data sets.
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks:
# - A
# 
# To Do & Ideas:
# 1. B
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 0.1.000 | 01/03/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning

# Image Processing
import skimage as ski
import imagesize


# Miscellaneous
import os
from platform import python_version, system
import random
# import warnings


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Typing
from typing import Callable, List, Tuple, Union

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

figIdx = 0

# %% Constants

PROJECT_NAME     = 'FixelCourses'
DATA_FOLDER_PATH = 'DataSets'
BASE_FOLDER      = os.getcwd()[:len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1]))]

L_IMG_EXT = ['.png', '.jpeg', '.jpg']


# %% Project Packages


# %% Auxiliary Functions


# %% Parameters

# Data
projectName = 'ships-aerial-images'

# Image
maxSize = 640 #<! [Pixel]


# %% Loading / Generating Data


# %% Analyze Data

projectDataFolder = os.path.join(BASE_FOLDER, DATA_FOLDER_PATH, projectName)
lFile             = os.listdir(projectDataFolder)
numRemImg         = 0 #<! Number of removed images

for ii, itmName in enumerate(lFile):
    itmPath = os.path.join(projectDataFolder, itmName)
    if os.path.isdir(itmPath):
        # Verify a YOLO Folder
        labelsFolderPath = os.path.join(itmPath, 'labels')
        imagesFolderPath = os.path.join(itmPath, 'images')
        if not (os.path.isdir(labelsFolderPath) and os.path.isdir(imagesFolderPath)):
            raise RuntimeError(f'Folder {itmPath} does not follow YOLO dataset structure')
        print(f'Processing {itmName}')
        lImgFiles = os.listdir(imagesFolderPath)
        numImgs = len(lImgFiles)
        for jj, imgFileName in enumerate(lImgFiles): 
            imgName, imgExt = os.path.splitext(imgFileName)
            if not (imgExt in L_IMG_EXT):
                continue
            print(f'Processing image #{(jj + 1):04d} out of {numImgs}')
            imgPath = os.path.join(imagesFolderPath, imgFileName)
            imgWidth, imgHeight = imagesize.get(imgPath)
            if (imgWidth > maxSize) or (imgHeight > maxSize):
                print(f'Removing {imgFileName}')
                # Delete Image and Label
                labelPath = os.path.join(labelsFolderPath, imgName + '.txt')
                os.remove(imgPath)
                os.remove(labelPath)
                # Debug code (Instead of `os.remove()` lines):
                # if not os.path.isfile(imgPath):
                #     raise RuntimeError(f'The file {imgPath} can not be found')
                # if not os.path.isfile(labelPath):
                #     raise RuntimeError(f'The file {labelPath} can not be found')
                numRemImg += 1

print(f'Removed {numRemImg} images')


# %% Plot Results


# %%
