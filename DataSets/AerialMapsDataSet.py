# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Satellite Aerial Maps Dataset
# Arranges a dataset for training based on the aerial maps in Cycle GAN dataset.
# The dataset is available at: [CycleGAN Datasets](https://efrosgans.eecs.berkeley.edu/cyclegan/datasets).
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 09/08/2026 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
# import scipy as sp
# import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random
import requests
import shutil
import zipfile

# Visualization
# import matplotlib.pyplot as plt


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


# %% Local Packages


# %% Auxiliary Functions

def DownloadFileUrl( fileUrl: str, outFile: str ) -> None:
    
    r = requests.get(fileUrl, allow_redirects = True)
    open(outFile, 'wb').write(r.content)


# %% Parameters

datasetName     = 'SatAerialToMap'
trainFolderName = 'Train'
valFolderName   = 'Validation'
rawDataUrl      = 'https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/maps.zip'


# %% Download

rawFilePath = os.path.join(DATA_FOLDER_PATH, f'{datasetName}.zip')
if not os.path.isfile(rawFilePath):
    DownloadFileUrl(rawDataUrl, rawFilePath)


# %% Unzip and Copy to Folders

# The Maps in `maps.zip` are stored:
# - maps
#  - test
#  - testA
#  - testB
#  - train
#    - resize
#  - trainA
#  - trainB
#  - val
#  - valA
#  - valB
#
# In each folder the images are named `1.jpg`, `2.jpg`, `3.jpg`, etc.
# The script will copy the images to the following structure:
# - SatAerialToMap
#  - Train <- (maps/train, maps/val)
#  - Validation (maps/test)
# The script will handle the collide in names by renaming the images to `00001.jpg`, `00002.jpg`, etc.

trainFolderPath = os.path.join(DATA_FOLDER_PATH, datasetName, trainFolderName)
valFolderPath   = os.path.join(DATA_FOLDER_PATH, datasetName, valFolderName)

if os.path.isdir(trainFolderPath):
    # Remove existing folder
    shutil.rmtree(trainFolderPath)

if os.path.isdir(valFolderPath):
    # Remove existing folder
    shutil.rmtree(valFolderPath)

os.makedirs(trainFolderPath, exist_ok = True)
os.makedirs(valFolderPath, exist_ok = True)

zipFilePath = os.path.join(DATA_FOLDER_PATH, f'{datasetName}.zip')
with zipfile.ZipFile(zipFilePath, 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DATA_FOLDER_PATH, datasetName))

# Rename files in `<DATA_FOLDER_PATH/datasetName/maps/train` to the pattern `00001.jpg`, `00002.jpg`, etc.
# Rename files in `<DATA_FOLDER_PATH/datasetName/maps/val` to continue after the last index in the train folder.
# Copy files from `maps/train`, `maps/val`, and `maps/test` to the corresponding folders.

mapsFolderPath = os.path.join(DATA_FOLDER_PATH, datasetName, 'maps')
trainImgIdx = 1

for sourceFolderName in ('train', 'val'):
    sourceFolderPath = os.path.join(mapsFolderPath, sourceFolderName)
    fileNameList = sorted(
        (fileName for fileName in os.listdir(sourceFolderPath) if fileName.lower().endswith('.jpg')),
        key = lambda fileName: int(os.path.splitext(fileName)[0]),
    )

    for fileName in fileNameList:
        sourceFilePath = os.path.join(sourceFolderPath, fileName)
        targetFilePath = os.path.join(trainFolderPath, f'{trainImgIdx:05d}.jpg')
        shutil.copy2(sourceFilePath, targetFilePath)
        trainImgIdx += 1

# Rename files in `<DATA_FOLDER_PATH/datasetName/maps/test` to the pattern `00001.jpg`, `00002.jpg`, etc.
# Copy files from `maps/test` to the corresponding folders.

testFolderPath = os.path.join(mapsFolderPath, 'test')
fileNameList = sorted(
    (fileName for fileName in os.listdir(testFolderPath) if fileName.lower().endswith('.jpg')),
    key = lambda fileName: int(os.path.splitext(fileName)[0]),
)

for valImgIdx, fileName in enumerate(fileNameList, start = 1):
    sourceFilePath = os.path.join(testFolderPath, fileName)
    targetFilePath = os.path.join(valFolderPath, f'{valImgIdx:05d}.jpg')
    shutil.copy2(sourceFilePath, targetFilePath)

# %% Delete All RAW Files

if os.path.isfile(rawFilePath):
    os.remove(rawFilePath)

zipOutputFolderPath = os.path.join(DATA_FOLDER_PATH, datasetName, 'maps')

if os.path.isdir(zipOutputFolderPath):
    shutil.rmtree(zipOutputFolderPath)

# %%
