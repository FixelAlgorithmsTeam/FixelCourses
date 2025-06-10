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

# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration


# %% Constants

# Matplotlib default color palette
L_MATPLOTLIB_COLOR = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# %% Auxiliary Classes


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

