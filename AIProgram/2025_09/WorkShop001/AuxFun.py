# %% Packages

# Python STD
from enum import auto, Enum, unique

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union
from numpy.typing import NDArray

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous

# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration


# %% Constants

@unique
class BBoxFormat(Enum):
    # Bounding Box Format, See https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations
    COCO        = auto()
    PASCAL_VOC  = auto()
    YOLO        = auto()

# Matplotlib default color palette
L_MATPLOTLIB_COLOR = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# %% Auxiliary Classes


# %% Auxiliary Functions

def GenTileImg( mI: NDArray, lTileSize: Union[List[int], Tuple[int]], lRow: List, lCol: List ) -> List:

    lTile = []
    for rowIdx in lRow:
        for colIdx in lCol:
            lTile.append(mI[rowIdx:(rowIdx + lTileSize[0]), colIdx:(colIdx + lTileSize[1]), :])
    
    return lTile

def ConvertPascalVocYolo( vBoxVoc: NDArray, imgW: int, imgH: int ) -> NDArray:
    # vBoxVoc: `[xmin, ymin, xmax, ymax]`

    boxCenterX = (vBoxVoc[0] + vBoxVoc[2]) / 2.0
    boxCenterY = (vBoxVoc[1] + vBoxVoc[3]) / 2.0

    boxWidth  = vBoxVoc[2] - vBoxVoc[0]
    boxHeight = vBoxVoc[3] - vBoxVoc[1]

    return np.array([boxCenterX / imgW, boxCenterY / imgH, boxWidth / imgW, boxHeight / imgH])

def ConvertRectPascalVoc( mBox: NDArray ) -> NDArray:
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

def ConvertBBoxFormat( vBox: NDArray, tuImgSize: Tuple[int, int], boxFormatIn: BBoxFormat, boxFormatOut: BBoxFormat ) -> NDArray:
    # tuImgSize = (numRows, numCols) <=> (imgHeight, imgWidth)

    vB = vBox.copy()
    
    # COCO = [xMin, yMin, boxWidth, boxHeight]
    if ((boxFormatIn == BBoxFormat.COCO) and (boxFormatOut == BBoxFormat.PASCAL_VOC)):
        vB[2] += vB[0] #<! xMax = Width + xMin
        vB[3] += vB[1] #<! yMax = Height + yMin
    elif ((boxFormatIn == BBoxFormat.COCO) and (boxFormatOut == BBoxFormat.YOLO)):
        vB[0] += (vB[2] / 2)  #<! xCenter = xMin + (boxWidth / 2)
        vB[1] += (vB[3] / 2)  #<! yCenter = yMin + (boxHeight / 2)
        vB[0] /= tuImgSize[1] #<! xCenter / imgWidth
        vB[1] /= tuImgSize[0] #<! yCenter / imgHeight
        vB[2] /= tuImgSize[1] #<! boxWidth / imgWidth
        vB[3] /= tuImgSize[0] #<! boxHeight / imgHeight
    
    # PASCAL_VOC = [xMin, yMin, xMax, yMax]
    elif ((boxFormatIn == BBoxFormat.PASCAL_VOC) and (boxFormatOut == BBoxFormat.COCO)):
        vB[2] -= vB[0] #<! boxWidth  = xMax - xMin
        vB[3] -= vB[1] #<! boxHeight = yMax - yMin
    elif ((boxFormatIn == BBoxFormat.PASCAL_VOC) and (boxFormatOut == BBoxFormat.YOLO)):
        vB[0] = (vB[0] + vB[2]) / 2                 #<! xCenter = (xMin + xMax) / 2
        vB[1] = (vB[1] + vB[3]) / 2                 #<! yCenter = (yMin + yMax) / 2
        vB[0] /= tuImgSize[1]                       #<! xCenter / imgWidth
        vB[1] /= tuImgSize[0]                       #<! yCenter / imgHeight
        vB[2] = (vBox[2] - vBox[0]) / tuImgSize[1]  #<! boxWidth = (xMax - xMin) / imgWidth
        vB[3] = (vBox[3] - vBox[1]) / tuImgSize[0]  #<! boxHeight = (YMax - yMin) / imgHeight
    
    # YOLO = [xCenter, yCenter, boxWidth, boxHeight] (Normalized)
    elif ((boxFormatIn == BBoxFormat.YOLO) and (boxFormatOut == BBoxFormat.COCO)):
        vB[0] -= (vB[2] / 2.0) #!< xMin = xCenter - (boxWidth / 2)
        vB[1] -= (vB[3] / 2.0) #!< yMin = yCenter - (boxHeight / 2)
        vB[0] *= tuImgSize[1]  #<! xMin * imgWidth
        vB[1] *= tuImgSize[0]  #<! yMin * imgHeight
        vB[2] *= tuImgSize[1]  #<! boxWidth * imgWidth
        vB[3] *= tuImgSize[0]  #<! boxHeight * imgHeight
    elif ((boxFormatIn == BBoxFormat.YOLO) and (boxFormatOut == BBoxFormat.PASCAL_VOC)):
        vB[0] -= (vB[2] / 2.0) #!< xMin = xCenter - (boxWidth / 2)
        vB[1] -= (vB[3] / 2.0) #!< yMin = yCenter - (boxHeight / 2)
        vB[2] += vB[0]         #<! xMax = boxWidth + xMin
        vB[3] += vB[1]         #<! yMax = boxHeight + yMin
        vB[0] *= tuImgSize[1]  #<! xMin * imgWidth
        vB[1] *= tuImgSize[0]  #<! yMin * imgHeight
        vB[2] *= tuImgSize[1]  #<! xMax * imgWidth
        vB[3] *= tuImgSize[0]  #<! yMax * imgHeight
    
    return vB


def PlotBox( mI: NDArray, vLabel: Union[int, NDArray], mBox: NDArray, *, hA: Optional[plt.Axes] = None, lLabelText: Optional[List] = None ) -> plt.Axes:
    # Assumes data in YOLO Format: [x, y, w, h] (Center, Height, Width)
    
    if hA is None:
        dpi = 72
        numRows, numCols = mI.shape[:2]
        hF, hA = plt.subplots(figsize = (int(np.ceil(numCols / dpi) + 1), int(np.ceil(numRows / dpi) + 1)))
    
    hA.imshow(mI, aspect = 'auto', extent = [0, 1, 1, 0]) #<! "Normalized Image"
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

def PlotBBox( hA: plt.Axes, boxLabel: int, vBox: NDArray, labelText: str = '_' ) -> plt.Axes:
    # Assumes data in YOLO Format
    # Legend Text: https://stackoverflow.com/questions/24680981

    edgeColor = hA._get_lines.get_next_color()

    rectPatch = Rectangle((vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2)), vBox[2], vBox[3], linewidth = 2, edgecolor = edgeColor, facecolor = (0, 0, 0, 0), label = labelText) #<! Requires the alpha component in the face color
    hA.add_patch(rectPatch)
    hA.text(vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2), s = boxLabel, color = 'w', verticalalignment = 'bottom', bbox = {'color': edgeColor}, fontdict = {'size': 16})
    hA.plot(vBox[0], vBox[1], 'x', mew = 5, ms = 10, color = edgeColor)

    return hA

