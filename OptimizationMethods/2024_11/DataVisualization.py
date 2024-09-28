
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

# Machine Learning

# Image Processing / Computer Vision

# Optimization

# Auxiliary

# Visualization
import distinctipy
import matplotlib.colors
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Miscellaneous
from enum import auto, Enum, unique

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# Course Packages


# Constants
FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2


def Plot2DFun( vX: np.ndarray, vY: np.ndarray, mZ: np.ndarray, /, *, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize, subplot_kw = {'projection': '3d'})
    else:
        hF = hA.get_figure()

    mX, mY = np.meshgrid(vX, vY)
    
    hSurf = hA.plot_surface(mX, mY, mZ, cmap = 'viridis')

    hA.set_title('')
    hA.set_xlabel('x')
    hA.set_ylabel('y')
    hA.set_zlabel('f(x, y)')

    # hF.colorbar(hSurf, shrink = 0.5, aspect = 5)
    hF.colorbar(hSurf, fraction = 0.05)

    return hA

def PlotBinaryClassData( mX: np.ndarray, vY: np.ndarray, /, *, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF, 
                        elmSize: int = ELM_SIZE_DEF, classColor: Tuple[str, str] = CLASS_COLOR, axisTitle: Optional[str] = None ) -> plt.Axes:
    """
    Plots binary 2D data as a scatter plot.
    Input:
        mX          - Matrix (numSamples, 2) of the data points.
        vY          - Vector (numSamples) labels of the data (2 Distinct values only).
    Output:
        hA          - Axes handler the scatter was drawn on.
    """

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize)
    else:
        hF = hA.get_figure()
    
    vC, vN = np.unique(vY, return_counts = True)

    numClass = len(vC)
    if (len(vC) != 2):
        raise ValueError(f'The input data is not binary, the number of classes is: {numClass}')

    vIdx0 = vY == vC[0]
    vIdx1 = vY == vC[1] #<! Basically ~vIdx0

    hA.scatter(mX[vIdx0, 0], mX[vIdx0, 1], s = elmSize, color = classColor[0], edgecolor = 'k', label = f'$C_\u007b {vC[0]} \u007d$')
    hA.scatter(mX[vIdx1, 0], mX[vIdx1, 1], s = elmSize, color = classColor[1], edgecolor = 'k', label = f'$C_\u007b {vC[1]} \u007d$')
    hA.axvline(x = 0, color = 'k')
    hA.axhline(y = 0, color = 'k')
    hA.axis('equal')
    if axisTitle is not None:
        hA.set_title(axisTitle)
    hA.legend()
    
    return hA
    

