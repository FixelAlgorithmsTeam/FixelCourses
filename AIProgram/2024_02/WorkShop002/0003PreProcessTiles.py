# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Object Detection Workshop
# Data Pre Processing:
#  1. Tiled Inference.
#  2. Labels Conversion.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.001 | 07/07/2024 | Royi Avital | Fixed issue with LabelMe which does not guarantee `[x_min, y_min, x_max, y_max]`         |
# |         |            | Royi Avital | Disabled the contrast check in SciKit Image `imsave()`                                   |
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
TILES_FOLDER_NAME = 'Tiles'
DRIVE_FOLDER_URL  = 'https://drive.google.com/drive/u/2/folders/1wxKIDN777K8kQ4UhJMu5csSbTVXhG7G9'

D_CLS = {'Ball': 0, 'Referee': 1, 'Referre': 1}
L_CLS = ['Ball', 'Referee']


# %% Local Packages


# %% Auxiliary Functions

def GenTileImg( mI: np.ndarray, lTileSize: Union[List[int], Tuple[int]], lRow: List, lCol: List ) -> List:

    lTile = []
    for rowIdx in lRow:
        for colIdx in lCol:
            lTile.append(mI[rowIdx:(rowIdx + lTileSize[0]), colIdx:(colIdx + lTileSize[1]), :])
    
    return lTile

def ConvertPascalVocYolo( vBoxVoc: np.ndarray, imgW: int, imgH: int ) -> np.ndarray:

    boxCenterX = (vBoxVoc[0] + vBoxVoc[2]) / 2.0
    boxCenterY = (vBoxVoc[1] + vBoxVoc[3]) / 2.0

    boxWidth  = vBoxVoc[2] - vBoxVoc[0]
    boxHeight = vBoxVoc[3] - vBoxVoc[1]

    return np.array([boxCenterX / imgW, boxCenterY / imgH, boxWidth / imgW, boxHeight / imgH])

def ConvertRectPascalVoc( mBox: np.ndarray ) -> np.ndarray:
    # https://github.com/labelmeai/labelme/issues/552
    # LabelMe Doesn't guarantee [x_min, y_min, x_max, y_max] (Works like the actual annotation).
    # This function convert it into such case.
    
    mB = np.copy(mBox)
    xMin = np.min(mBox[:, 0], axis = 0)
    xMax = np.max(mBox[:, 0], axis = 0)
    yMin = np.min(mBox[:, 1], axis = 0)
    yMax = np.max(mBox[:, 1], axis = 0)

    mB[0, 0] = xMin
    mB[0, 1] = yMin
    mB[1, 0] = xMax
    mB[1, 1] = yMax
    
    return mB


def PlotBox( mI: np.ndarray, vLabel: Union[int, np.ndarray], mBox: np.ndarray, *, hA: Optional[plt.Axes] = None, lLabelText: Optional[List] = None ) -> plt.Axes:
    # Assumes data in YOLO Format: [x, y, w, h] (Center, Height, Width)
    
    if hA is None:
        dpi = 72
        numRows, numCols = mI.shape[:2]
        hF, hA = plt.subplots(figsize = (int(np.ceil(numCols / dpi) + 1), int(np.ceil(numRows / dpi) + 1)))
    
    hA.imshow(mI, extent = [0, 1, 1, 0]) #<! "Normalized Image"
    hA.grid(False)

    mBox = np.atleast_2d(mBox)
    vLabel = np.atleast_1d(vLabel)
    numObj = mBox.shape[0]
    for ii in range(numObj):
        if lLabelText is not None:
            labelText = lLabelText[ii]
        else:
            labelText = '_'
        PlotBBox(hA, vLabel[ii], mBox[ii], labelText)

    return hA

def PlotBBox( hA: plt.Axes, boxLabel: int, vBox: np.ndarray, labelText: str = '_' ) -> plt.Axes:
    # Assumes data in YOLO Format
    # Legend Text: https://stackoverflow.com/questions/24680981

    edgeColor = hA._get_lines.get_next_color()

    rectPatch = Rectangle((vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2)), vBox[2], vBox[3], linewidth = 2, edgecolor = edgeColor, facecolor = (0, 0, 0, 0), label = labelText) #<! Requires the alpha component in the face color
    hA.add_patch(rectPatch)
    hA.text(vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2), s = boxLabel, color = 'w', verticalalignment = 'bottom', bbox = {'color': edgeColor}, fontdict = {'size': 16})
    hA.plot(vBox[0], vBox[1], 'x', mew = 5, ms = 10, color = edgeColor)

    return hA


# %% Parameters

imgFileExt  = 'png'
lblFileExt  = 'json'
yoloFileExt = 'txt'

lTileSize = [640, 640]
# TODO: Make the grid 3 x 5 (As the referee can be ~120 pixels high)
lRow      = [0, 440] #<! Row to start at
lCol      = [0, 320, 640, 960, 1280] #<! Column to start at


# %% Load / Generate Data

dataFolderPath  = os.path.join(os.getcwd(), DATA_FOLDER_NAME)
tilesFolderPath = os.path.join(dataFolderPath, TILES_FOLDER_NAME)

# Clean Tiles folder
if os.path.isdir(tilesFolderPath):
    shutil.rmtree(tilesFolderPath) 

os.mkdir(tilesFolderPath)

# List of Files
# Valid file: Ends with `png` and has annotation file (Same name, JSON)
lFile = []
for ii, fullFileName in enumerate(os.listdir(dataFolderPath)):
    fileName, fileExt = os.path.splitext(fullFileName)
    if ((imgFileExt in fileExt) and (os.path.isfile(os.path.join(dataFolderPath, fileName + '.' + lblFileExt)))):
        lFile.append(fullFileName)


# %% Generate Tiles with Annotations

numTiles = len(lRow) * len(lCol)

for ii, fullFileName in enumerate(lFile):
    fileName, fileExt = os.path.splitext(fullFileName)
    jsonFileName = fileName + '.' + lblFileExt
    jsonFilePath = os.path.join(dataFolderPath, jsonFileName)
    with open(jsonFilePath) as jsonfile:
        dJson = json.load(jsonfile)
    
    numLabels = len(dJson['shapes'])
    imgH = dJson['imageHeight']
    imgW = dJson['imageWidth']

    # Iterating over tiles
    tileIdx  = 0
    for rowIdx in lRow:
        for colIdx in lCol:
            yoloFile = False
            imgFile  = False
            tileIdx += 1
            mTile = np.array([[colIdx, rowIdx], [colIdx + lTileSize[1], rowIdx + lTileSize[0]]]) #<! [[xmin, ymin], [xmax, ymax]]
            for dLbl in dJson['shapes']:
                lblClsStr = dLbl['label']
                lblCls = D_CLS[lblClsStr]
                mBox = np.array(dLbl['points']) #<! [[xmin, ymin], [xmax, ymax]]
                mBox = ConvertRectPascalVoc(mBox)
                tileFileName = fileName + f'Tile{tileIdx:03d}'

                boxInTile = np.all(np.logical_and( mBox[0] >= mTile[0], mBox[1] < mTile[1] ))
                if boxInTile:
                    tileFileNameLbl = tileFileName + '.' + yoloFileExt
                    tileFileNameImg = tileFileName + '.' + imgFileExt
                    mBoxTile = mBox - np.array([colIdx, rowIdx])[None, :] #<! Coordinates in Tile
                    vBoxYolo = ConvertPascalVocYolo(mBoxTile.flat, lTileSize[1], lTileSize[0]) #<! TODO: Check for negative values
                    if np.any(vBoxYolo < 0.0):
                        print(fullFileName)
                        print(np.ravel(mBox))
                        print((colIdx, rowIdx))
                    with open(os.path.join(tilesFolderPath, tileFileNameLbl), 'a') as hFile:
                        print(f'{lblCls} {vBoxYolo[0]:0.5f} {vBoxYolo[1]:0.5f} {vBoxYolo[2]:0.5f} {vBoxYolo[3]:0.5f}', file = hFile)
                    
                    if not imgFile:
                        imgFile = True
                        mI = ski.io.imread(os.path.join(dataFolderPath, fullFileName))
                        mT = mI[rowIdx:(rowIdx + lTileSize[0]), colIdx:(colIdx + lTileSize[1]), :]
                        ski.io.imsave(os.path.join(tilesFolderPath, tileFileNameImg), mT, check_contrast = False)


# %% Display Results

lFile = [fileName for fileName in os.listdir(tilesFolderPath) if fileName.endswith(imgFileExt)]
numFiles = len(lFile)

imgIdx = random.randrange(numFiles)

imgFullName = lFile[imgIdx]
fileName, fileExt = os.path.splitext(imgFullName)
yoloLblFullName = fileName + '.' + yoloFileExt

mI = ski.io.imread(os.path.join(tilesFolderPath, imgFullName))

with open(os.path.join(tilesFolderPath, yoloLblFullName), 'r') as hFile:
    lLine = hFile.read().splitlines()

numLbls = len(lLine)
vLbl = np.zeros(shape = numLbls, dtype = np.int_)
mBox = np.zeros(shape = (numLbls, 4))
for ii, lineStr in enumerate(lLine):
    lA = lineStr.split() #<! Split by spaces
    lB = [float(num) for num in lA]
    vLbl[ii] = int(lA[0])
    mBox[ii, :] = lB[1:]

lLbl = [L_CLS[lblIdx] for lblIdx in vLbl]

PlotBox(mI, vLbl, mBox)

# %%
