# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Download [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 26/01/2025 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Python Standard Library
import os
from platform import python_version
import random
import shutil

# Miscellaneous
import onedrivedownloader

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_NAME     = 'FixelCourses'
DATA_FOLDER_NAME = 'DataSets'
BASE_FOLDER_PATH = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)

DATA_SET_FILE_NAME      = 'IntelImgCls.zip'
DATA_SET_FOLDER_NAME    = 'IntelImgCls'

D_CLASSES  = {0: 'Buildings', 1: 'Forest', 2: 'Glacier', 3: 'Mountain', 4: 'Sea', 5: 'Street'}
L_CLASSES  = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']


# %% Local Packages


# %% Auxiliary Functions


# %% Parameters

# Link to RAW Data (Not processed into folder structure)
fileUrl = 'https://technionmail-my.sharepoint.com/:u:/g/personal/royia_technion_ac_il/EXnOy43-NqZJic4PXql9x8sB8lfoMKPponmp0zxeXnQAsw?e=MsYu3i' #<! OneDrive URL


# %% [markdown]

## Generate / Load Data

# This notebook use the [Intel Image Classification Data Set](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).  
# The data set is composed of 6 classes: `Buildings`, `Forest`, `Glacier`, `Mountain`, `Sea`, `Street`.
# 
# The following code will arrange the data in a manner compatible with PyTorch's [`ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).
# 
# * <font color='brown'>(**#**)</font> The data set originally appeared on [Analytics Vidhya - Practice Problem: Intel Scene Classification Challenge](https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe).
# * <font color='brown'>(**#**)</font> Some of the images are not `150x150x3` hence they should be handled.
# * <font color='brown'>(**#**)</font> Some of the images are not labeled correctly (See discussions on Kaggle).
# 
# ### Downloading the Data
# 
# The data should be downloaded automatically.  
# **In case of an issue**, it can be downloaded manually:
# 
# 1. Download the Zip file `archive.zip` from [Intel Image Classification Data Set](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
# 2. Copy / Move the file into `AIProgram/<YYYY_MM>/Data` folder.
# 3. Rename the file to match `DATA_SET_FILE_NAME`.
# 4. Comment the line `onedrivedownloader.download(fileUrl, os.path.join(DATA_FOLDER_PATH, DATA_SET_FILE_NAME), unzip = False)` and run the cell below.

# %% Download Data

# Arrange Data for Image Folder
# Assumes `archive.zip` in `./Data`

DATA_FOLDER_PATH  = 1

dataSetPath = os.path.join(DATA_FOLDER_PATH, DATA_SET_FOLDER_NAME)
if not os.path.isdir(dataSetPath):
    os.mkdir(dataSetPath)
lFiles = os.listdir(dataSetPath)

onedrivedownloader.download(fileUrl, os.path.join(DATA_FOLDER_PATH, DATA_SET_FILE_NAME), unzip = False)

if '.processed' not in lFiles: #<! Run only once
    os.makedirs(os.path.join(dataSetPath, 'TMP'), exist_ok = True)
    os.makedirs(os.path.join(dataSetPath, 'Test'), exist_ok = True)
    for clsName in L_CLASSES:
        os.makedirs(os.path.join(dataSetPath, 'Train', clsName), exist_ok = True)
        os.makedirs(os.path.join(dataSetPath, 'Validation', clsName), exist_ok = True)
    
    shutil.unpack_archive(os.path.join(DATA_FOLDER_PATH, DATA_SET_FILE_NAME), os.path.join(dataSetPath, 'TMP'))

    for dirPath, lSubDir, lF in os.walk(os.path.join(dataSetPath, 'TMP')):
        if len(lF) > 0:
            if 'test' in dirPath:
                dstPath = os.path.join(dataSetPath, 'Validation')
            elif 'train' in dirPath:
                dstPath = os.path.join(dataSetPath, 'Train')
            else:
                dstPath = os.path.join(dataSetPath, 'Test')
            
            if 'buildings' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Buildings'))
            elif 'forest' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Forest'))
            elif 'glacier' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Glacier'))
            elif 'mountain' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Mountain'))
            elif 'sea' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Sea'))
            elif 'street' in dirPath:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), os.path.join(dstPath, 'Street'))
            else:
                for fileName in lF:
                    shutil.move(os.path.join(dirPath, fileName), dstPath)
    
    shutil.rmtree(os.path.join(dataSetPath, 'TMP'))

    hFile = open(os.path.join(dataSetPath, '.processed'), 'w')
    hFile.close()

