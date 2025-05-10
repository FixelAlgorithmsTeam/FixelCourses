
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


def Plot2DLinearClassifier( mX: np.ndarray, vY: np.ndarray, vW: np.ndarray, mX1: np.ndarray, mX2: np.ndarray, hA: plt.Axes ) -> None:
    """
    Plots a binary 2D classifier.
    Input:
        mX          - Matrix (3, numSamples) of the data points.
        vY          - Vector (numSamples) labels of the data.
        vW          - Vector (3) Parameters of the classifier.
        mX1         - Matrix (numGridPtsX1, numGridPtsX2) of the grid points (1st variable).
        mX2         - Matrix (numGridPtsX1, numGridPtsX2) of the grid points (2nd variable).
        hA          - Axes handler the scatter was drawn X☻on.
    Output:
    Remarks:
      - The model parameters `vW` match a matrix of the form: y_i = w_1 * x_1 + w_2 * x2 - w_0.
      - The matrices `mX1` and `mX2` are the result of `np.meshgrid()` of 2D grid.
    """
    b  = vW[0]
    vW = vW[1:]
    XX = np.column_stack([mX1.flatten(), mX2.flatten()])

    if (vW[1] == 0):
        vW[1]  = 1e-9
        vW[0] *= 1e9
        b     *= 1e9

    vZ = (XX @ vW - b) > 0
    ZZ = vZ.reshape(mX1.shape)
    
    vHatY    = np.sign(mX @ vW - b)
    accuracy = np.mean(vY == vHatY)

    axisTitle = r'$f_{{w},b} \left( {x} \right) = {sign} \left( {w}^{T} {x} - b \right)$' '\n' f'Accuracy = {accuracy:.2%}'

    PlotBinaryClassData(mX, vY, hA = hA, axisTitle = axisTitle)
    v = np.array([-2, 2])
    hA.grid(True)
    hA.plot(v, -(vW[0] / vW[1]) * v + (b / vW[1]), color = 'k', lw = 3)
    hA.arrow(0, 0, vW[0], vW[1], color = 'orange', width = 0.05)
    hA.axvline(x = 0, color = 'k', lw = 1)
    hA.axhline(y = 0, color = 'k', lw = 1)
    hA.contourf(mX1, mX2, ZZ, colors = CLASS_COLOR, alpha = 0.2, levels = [-0.5, 0.5, 1.5], zorder = 0)
    
    hA.set_xlim([-2, 2])
    hA.set_ylim([-2, 2])
    hA.set_xlabel('$x_1$')
    hA.set_ylabel('$x_2$')

    return

