# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Image Segmentation Workshop
# Downloads and arranges the The Oxford-IIIT Pet Dataset.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.1.000 | 10/06/2025 | Royi Avital | Converting `jpg` to `png` to overcome corrupted files                                    |
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
from PIL import Image

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random
# import warnings
import shutil
import urllib.request


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

L_IMG_FILE_EXT = ['.jpg', '.jpeg', '.png']

D_CLASSES  = {0: 'Red', 1: 'Green', 2: 'Blue'}
L_CLASSES  = ['R', 'G', 'B']
T_IMG_SIZE = (100, 100, 3)

DATA_FOLDER_NAME  = 'Data'
DATA_SET_FOLDER   = 'OxfordIIITPet'
ANN_FILE_URL      = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz'
IMG_FILE_URL      = 'https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz'


# %% Local Packages


# %% Auxiliary Functions

def DownloadUrl( fileUrl: str, fileName: str ) -> str:
    
    if not os.path.exists(fileName):
        urllib.request.urlretrieve(fileUrl, fileName)

    return fileName

def ExtFileName( fullFileName: str ) -> str:
    # Extracts the file name without the extension.

    fileName, fileExt = os.path.splitext(fullFileName)

    return fileName


# %% Parameters

annArchName = 'Annotations.tar.gz'
imgArchName = 'Images.tar.gz'

fileExt     = 'png'


# %% Load / Generate Data
# Will have:
# - Data
#    - OxfordIIITPet
#       - Annotations
#       - Images
# Torrent alternative (If fails): https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1

dataFolderPath = os.path.join(os.getcwd(), DATA_FOLDER_NAME)
os.makedirs(dataFolderPath, exist_ok = True)

annArchPath = os.path.join(dataFolderPath, annArchName)
imgArchPath = os.path.join(dataFolderPath, imgArchName)

if not os.path.isfile(annArchPath):
    print(f'Downloading annotation files to {dataFolderPath}')
    DownloadUrl(ANN_FILE_URL, annArchPath)

if not os.path.isfile(imgArchPath):
    print(f'Downloading image files to {dataFolderPath}')
    DownloadUrl(IMG_FILE_URL, imgArchPath)

annFolderPath = os.path.join(dataFolderPath, DATA_SET_FOLDER, 'Annotations')
imgFolderPath = os.path.join(dataFolderPath, DATA_SET_FOLDER, 'Images')

if (os.path.isdir(annFolderPath)):
    shutil.rmtree(annFolderPath)

if (os.path.isdir(imgFolderPath)):
    shutil.rmtree(imgFolderPath)

os.makedirs(annFolderPath, exist_ok = True)
os.makedirs(imgFolderPath, exist_ok = True)

print(f'Extracting annotation files to {annFolderPath}')
shutil.unpack_archive(annArchPath, annFolderPath)
print(f'Extracting image files to {imgFolderPath}')
shutil.unpack_archive(imgArchPath, imgFolderPath)

# Delete files
# The extraction creates: 'OxfordIIITPet\Annotations\annotations'
# Annotations are inside 'OxfordIIITPet\Annotations\annotations\trimaps'
# The rest of the files are not needed -> Deleted.
print(f'Removing invalid files and folders in {os.path.join(annFolderPath, 'annotations')}')
numFiles = len(os.listdir(os.path.join(annFolderPath, 'annotations')))
for ii, itmName in enumerate(os.listdir(os.path.join(annFolderPath, 'annotations'))):
    print(f'Processing {ii + 1}/{numFiles}: {itmName}')
    if not (itmName == 'trimaps'):
        if os.path.isfile(os.path.join(annFolderPath, 'annotations', itmName)):
            os.remove(os.path.join(annFolderPath, 'annotations', itmName))
        if os.path.isdir(os.path.join(annFolderPath, 'annotations', itmName)):
            shutil.rmtree(os.path.join(annFolderPath, 'annotations', itmName))

# Move valid annotations
# Copy from 'OxfordIIITPet\Annotations\annotations\trimaps' to 'OxfordIIITPet\Annotations'.
print(f'Copying annotation files from {os.path.join(annFolderPath, 'annotations')} to {annFolderPath}')
numFiles = len(os.listdir(os.path.join(annFolderPath, 'annotations', 'trimaps')))
for ii, itmName in enumerate(os.listdir(os.path.join(annFolderPath, 'annotations', 'trimaps'))):
    print(f'Processing {ii + 1}/{numFiles}: {itmName}')
    if not ('._' in itmName):
        itmPath = os.path.join(annFolderPath, 'annotations', 'trimaps', itmName)
        shutil.copy(itmPath, annFolderPath)

# Remove 'OxfordIIITPet\Annotations\annotations'
print(f'Removing {os.path.join(annFolderPath, 'annotations')} folder')
shutil.rmtree(os.path.join(annFolderPath, 'annotations'))

# Move Images
# Some images are invalid (Corrupted) `jpg` files (Issues with headers, etc...).
# Hence they are saved as 'png' files.
print(f'Converting (jpg -> png) image files from {os.path.join(imgFolderPath, 'images')} to {imgFolderPath}')
numFiles = len(os.listdir(os.path.join(imgFolderPath, 'images')))
for ii, itmName in enumerate(os.listdir(os.path.join(imgFolderPath, 'images'))):
    print(f'Processing {ii + 1}/{numFiles}: {itmName}')
    fileName, fileExt = os.path.splitext(itmName)
    if (fileExt in L_IMG_FILE_EXT):
        itmPath = os.path.join(imgFolderPath, 'images', itmName)
        # shutil.copy(itmPath, imgFolderPath) #<! Copy files
        # Convert files into `png`
        oImg = Image.open(itmPath)
        mI   = np.asarray(oImg)
        oImg = Image.fromarray(mI)
        oImg.save(os.path.join(imgFolderPath, f'{fileName}.png'), format = 'PNG')

# Remove 'OxfordIIITPet\Images\images'
print(f'Removing {os.path.join(imgFolderPath, 'images')} folder')
shutil.rmtree(os.path.join(imgFolderPath, 'images'))


# %% Validate Matching File Name
# If no matching, it will print

lAnnFileName = [ExtFileName(itmName) for itmName in os.listdir(annFolderPath)]
lImgFileName = [ExtFileName(itmName) for itmName in os.listdir(imgFolderPath)]

sAnnFileName = set(lAnnFileName)
sImgFileName = set(lImgFileName)

sDiff = sAnnFileName - sImgFileName
if (len(sDiff) > 0):
    print(f'Files in annotations yet not in images: {sDiff}')

sDiff = sImgFileName - sAnnFileName
if (len(sDiff) > 0):
    print(f'Files in images yet not in annotations: {sDiff}')


# %% Display Results

