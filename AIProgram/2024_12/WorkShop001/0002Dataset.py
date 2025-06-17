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

DATA_SET_FILE_NAME      = 'archive.zip'
DATA_SET_FOLDER_NAME    = 'IntelImgCls'

DATA_FOLDER_NAME  = 'Data'
TEST_FOLDER_NAME  = 'Test'
TRAIN_FOLDER_NAME = 'Train'
DRIVE_FOLDER_URL  = 'https://drive.google.com/drive/u/2/folders/1wxKIDN777K8kQ4UhJMu5csSbTVXhG7G9' #<! Labeled Data Set folder

D_CLS = {'Ball': 0, 'Referee': 1}
L_CLS = ['Ball', 'Referee']


# %% Local Packages


# %% Auxiliary Functions

from AuxFun import ConvertPascalVocYolo, ConvertRectPascalVoc, PlotBox, PlotBBox


# %% Parameters

# dataFileId = '1ocqIxXP--q0KXIylTKQIqTwQgw3ZsiX3' #<! Images + Labels (LabelMe)
dataFileId = '1LW3pX_dg8oQ2Q-hixeGo6AwNxtb_DPwg' #<! Images
imgFileExt = 'png'
lblFileExt = 'json' #<! LabelMe format


# %% [markdown]
#
# ## Generate / Load Data
# 
# This section is used to download the data set from Google Drive and extract it.      
# The dataset, composed of 300 images, will be extracted to `Data` folder.    
# Once the data is downloaded, it should be labeled using LabelMe or similar tool.    


# %% Load / Generate Data

dataFolderPath = os.path.join(os.getcwd(), DATA_FOLDER_NAME)

fileName = gdown.download(id = dataFileId)
if not (os.path.isdir(dataFolderPath)):
    os.mkdir(dataFolderPath)

# Move file, replaces if already exists (https://stackoverflow.com/a/8858026)
os.replace(fileName, os.path.join(dataFolderPath, fileName))

# Unpack the archive, the images, into the data folder
shutil.unpack_archive(os.path.join(dataFolderPath, fileName), dataFolderPath)


# %% [markdown]
#
# ## Label Data
# 
# Label the data as following:
# - The Classes
#   - Ball - The game ball.
#   - Referee - The referee.
#
# Use the policies as defined in the slides.
# At the end of this file the `Data` folder should contain a `.json` file per image.


# %% Display Results

# Show a single image with its labels
lFile    = [fileName for fileName in os.listdir(dataFolderPath) if fileName.endswith(imgFileExt)] #<! Assuming all images are labeled
numFiles = len(lFile)

imgIdx = random.randrange(numFiles)
# imgIdx = 224 #<! Used in the slides

imgFullName          = lFile[imgIdx]
fileName, imgFileExt = os.path.splitext(imgFullName)
jsonFullName         = fileName + '.' + lblFileExt

# Read image
mI = ski.io.imread(os.path.join(dataFolderPath, imgFullName))
# Read labels (LabelMe format -> Dictionary)
with open(os.path.join(dataFolderPath, jsonFullName)) as jsonfile:
        dJson = json.load(jsonfile)
    
numObj = len(dJson['shapes'])
imgH = dJson['imageHeight']
imgW = dJson['imageWidth']

lLbl = []
vLbl = np.zeros(shape = numObj, dtype = np.int_)
mBox = np.zeros(shape = (numObj, 4)) #<! YOLO Style (To match `PlotBox`)
for ii, dShape in enumerate(dJson['shapes']):
    lLbl.append(dShape['label'])
    vLbl[ii] = D_CLS[dShape['label']]
    # The LabelMe format (List of Lists): `[[xmin, ymin], [xmax, ymax]]` (Yet the order is not guaranteed)
    # It is similar to the PASCAL VOC format (`[xmin, ymin, xmax, ymax]`)
    mB = ConvertRectPascalVoc(np.array(dShape['points'])) #<! Guarantees the order
    mBox[ii, :] = ConvertPascalVocYolo(mB.flat, imgW, imgH) #<! TODO: Check for negative values

# lLbl = [L_CLS[lblIdx] for lblIdx in vLbl]

hA = PlotBox(mI, vLbl, mBox, lLabelText = lLbl)
hA.set_title(f'Image: {imgIdx}')
hA.legend();


# %%
