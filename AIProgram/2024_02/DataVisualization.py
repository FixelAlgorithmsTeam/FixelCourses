
# Python STD
# import enum
import math

# Data
import numpy as np
import pandas as pd
import scipy as sp

# Machine Learning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Image Processing / Computer Vision

# Optimization

# Auxiliary
from numba import jit, njit

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
from DataManipulation import BBoxFormat

# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

# Constants
FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2

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
    b = vW[0]
    vW = vW[1:]
    XX = np.column_stack([mX1.flatten(), mX2.flatten()])

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

def PlotMnistImages( mX: np.ndarray, vY: np.ndarray, numRows: int, numCols: Optional[int] = None, tuImgSize: Tuple = (28, 28), randomChoice: bool = True, lClasses: Optional[List] = None, hF: Optional[plt.Figure] = None ) -> plt.Figure:

    numSamples  = mX.shape[0]
    numPx       = mX.shape[1]

    if numCols is None:
        numCols = numRows

    tFigSize = (numCols * 3, numRows * 3)

    if hF is None:
        hF, hA = plt.subplots(numRows, numCols, figsize = tFigSize)
    else:
        hA = hF.axis
    
    hA = np.atleast_1d(hA) #<! To support numImg = 1
    hA = hA.flat
    
    for kk in range(numRows * numCols):
        idx = np.random.choice(numSamples) if randomChoice else kk
        mI  = np.reshape(mX[idx, :], tuImgSize)
    
        # hA[kk].imshow(mI.clip(0, 1), cmap = 'gray')
        if len(tuImgSize) == 2:
            hA[kk].imshow(mI, cmap = 'gray')
        elif len(tuImgSize) == 3:
            hA[kk].imshow(mI)
        else:
            raise ValueError(f'The length of the image size tuple is {len(tuImgSize)} which is not supported')
        hA[kk].tick_params(axis = 'both', left = False, top = False, right = False, bottom = False, 
                           labelleft = False, labeltop = False, labelright = False, labelbottom = False)
        if lClasses is None:
            hA[kk].set_title(f'Index = {idx}, Label = {vY[idx]}')
        else:
            hA[kk].set_title(f'Index = {idx}, Label = {lClasses[vY[idx]]}')
    
    return hF

def PlotLabelsHistogram( vY: np.ndarray, hA: Optional[plt.Axes] = None, lClass: Optional[List] = None, xLabelRot: Optional[int] = None ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = (8, 6))
    
    vLabels, vCounts = np.unique(vY, return_counts = True)

    hA.bar(vLabels, vCounts, width = 0.9, align = 'center')
    hA.set_title('Histogram of Classes / Labels')
    hA.set_xlabel('Class')
    hA.set_xticks(vLabels, [f'{labelVal}' for labelVal in vLabels])
    hA.set_ylabel('Count')
    if lClass is not None:
        hA.set_xticklabels(lClass)
    
    if xLabelRot is not None:
        for xLabel in hA.get_xticklabels():
            xLabel.set_rotation(xLabelRot)

    return hA

def PlotConfusionMatrix(vY: np.ndarray, vYPred: np.ndarray, normMethod: str = None, hA: Optional[plt.Axes] = None, 
                        lLabels: Optional[List] = None, dScore: Optional[Dict] = None, titleStr: str = 'Confusion Matrix', 
                        xLabelRot: Optional[int] = None, valFormat: Optional[str] = None) -> Tuple[plt.Axes, np.ndarray]:

    # Calculation of Confusion Matrix
    mConfMat = confusion_matrix(vY, vYPred, normalize = normMethod)
    oConfMat = ConfusionMatrixDisplay(mConfMat, display_labels = lLabels)
    oConfMat = oConfMat.plot(ax = hA, values_format = valFormat)
    hA = oConfMat.ax_
    if dScore is not None:
        titleStr += ':'
        for scoreName, scoreVal in  dScore.items():
            titleStr += f' {scoreName} = {scoreVal:0.2},'
        titleStr = titleStr[:-1]
    hA.set_title(titleStr)
    hA.grid(False)
    if xLabelRot is not None:
        for xLabel in hA.get_xticklabels():
            xLabel.set_rotation(xLabelRot)

    return hA, mConfMat

def PlotDecisionBoundaryClosure( numGridPts: int, gridXMin: float, gridXMax: float, gridYMin: float, gridYMax: float, clsColors: Tuple = CLASS_COLOR, numDigits: int = 1 ) -> Callable:

    roundFctr = 10 ** numDigits
    
    # For equal axis
    minVal = np.floor(roundFctr * min(gridXMin, gridYMin)) / roundFctr
    maxVal = np.ceil(roundFctr * max(gridXMax, gridYMax)) / roundFctr
    vX1    = np.linspace(minVal, maxVal, numGridPts)
    vX2    = np.linspace(minVal, maxVal, numGridPts)
    
    mX1, mX2 = np.meshgrid(vX1, vX2)
    mX       = np.c_[mX1.ravel(), mX2.ravel()] #<! Features (2D)

    # A closure
    def PlotDecisionBoundary(hDecFun: Callable, hA: plt.Axes = None) -> plt.Axes:
        
        if hA is None:
            hF, hA = plt.subplots(figsize = (8, 6))

        mZ = hDecFun(mX)
        mZ = mZ.reshape(mX1.shape)

        # Assumes values {0, 1}
        hA.contourf(mX1, mX2, mZ, colors = clsColors, alpha = 0.3, levels = [-0.5, 0.5, 1.5])

        return hA

    return PlotDecisionBoundary

def PlotRegressionData( mX: np.ndarray, vY: np.ndarray, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF, elmSize: int = ELM_SIZE_DEF, classColor: Tuple[str, str] = CLASS_COLOR, axisTitle: Optional[str] = None ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize)
    else:
        hF = hA.get_figure()
    
    if np.ndim(mX) == 1:
        mX = np.reshape(mX, (mX.size, 1))

    numSamples = len(vY)
    numDim     = mX.shape[1]
    if (numDim > 2):
        raise ValueError(f'The features data must have at most 2 dimensions')
    
    # Work on 1D, Add support for 2D when needed
    # See https://matplotlib.org/stable/api/toolkits/mplot3d.html
    hA.scatter(mX[:, 0], vY, s = elmSize, color = classColor[0], edgecolor = 'k', label = f'Samples')
    hA.axvline(x = 0, color = 'k')
    hA.axhline(y = 0, color = 'k')
    hA.set_xlabel('${x}_{1}$')
    # hA.axis('equal')
    if axisTitle is not None:
        hA.set_title(axisTitle)
    hA.legend()
    
    return hA

def PlotRegressionResults( vY, vYPred, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF, lineWidth: int = LINE_WIDTH_DEF, elmSize: int = ELM_SIZE_DEF, classColor: Tuple[str, str] = CLASS_COLOR, axisTitle: Optional[str] = None ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize)
    else:
        hF = hA.get_figure()

    numSamples = len(vY)
    if (numSamples != len(vYPred)):
        raise ValueError(f'The inputs `vY` and `vYPred` must have the same number of elements')
    
    hA.plot(vY, vY, color = 'r', lw = lineWidth, label = 'Ground Truth')
    hA.scatter(vY, vYPred, s = elmSize, color = classColor[0], edgecolor = 'k', label = f'Estimation')
    hA.set_xlabel('Label Value')
    hA.set_ylabel('Prediction Value')
    # hA.axis('equal')
    if axisTitle is not None:
        hA.set_title(axisTitle)
    hA.legend()
    
    return hA

def PlotScatterData( mX: np.ndarray, vL: Optional[np.ndarray] = None, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF, markerSize: int = ELM_SIZE_DEF, edgeColor: int = EDGE_COLOR ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize)
    else:
        hF = hA.get_figure()
    
    numSamples = mX.shape[0]

    if vL is None:
        vL = np.zeros(numSamples)
    
    vU = np.unique(vL)
    numClusters = len(vU)

    for ii in range(numClusters):
        vIdx = vL == vU[ii]
        hA.scatter(mX[vIdx, 0], mX[vIdx, 1], s = markerSize, edgecolor = edgeColor, label = ii)
    
    hA.set_xlabel('${{x}}_{{1}}$')
    hA.set_ylabel('${{x}}_{{2}}$')
    hA.grid()
    hA.legend()

    return hA

def PlotScatterData3D( mX: np.ndarray, vL: Optional[np.ndarray] = None, vC: Optional[np.ndarray] = None, axesProjection: Optional[str] = '3d', hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = figSize, subplot_kw = {'projection': axesProjection})
    else:
        hF = hA.get_figure()
    
    if vL is not None:
        vU = np.unique(vL)
        numClusters = len(vU)
    else:
        vL = np.zeros(mX.shape[0])
        vU = np.zeros(1)
        numClusters = 1

    for ii in range(numClusters):
        vIdx = vL == vU[ii]
        if axesProjection == '3d':
            hA.scatter(mX[vIdx, 0], mX[vIdx, 1], mX[vIdx, 2], s = ELM_SIZE_DEF, c = vC, alpha = 1, edgecolor = EDGE_COLOR, label = ii)
        else:
            hA.scatter(mX[vIdx, 0], mX[vIdx, 1], s = ELM_SIZE_DEF, c = vC, alpha = 1, edgecolor = EDGE_COLOR, label = ii)
    
    hA.set_xlabel('${{x}}_{{1}}$')
    hA.set_ylabel('${{x}}_{{2}}$')
    if axesProjection == '3d':
        hA.set_zlabel('${{x}}_{{3}}$')
    hA.grid()
    hA.legend()

    return hA

def PlotDendrogram( dfX: Union[np.ndarray, pd.DataFrame], linkageMethod: str, valP: int, thrLvl: int, hA: Optional[plt.Axes] = None, figSize: Tuple[int, int] = FIG_SIZE_DEF ) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(1, 1, figsize = figSize)
    else:
        hF = hA.get_figure()

    mLinkage = sp.cluster.hierarchy.linkage(dfX, method = linkageMethod)
    sp.cluster.hierarchy.dendrogram(mLinkage, p = valP, truncate_mode = 'lastp', color_threshold = thrLvl, no_labels = True, ax = hA)
    hA.axhline(y = thrLvl, c = 'k', lw = 2, linestyle = '--')

    return hA

def PlotBox( mI: np.ndarray, boxLabel: int, vBox: np.ndarray, *, hA: Optional[plt.Axes] = None, boxScore: Optional[float] = None ) -> plt.Axes:
    # Assumes data in YOLO Format
    
    if hA is None:
        dpi = 72
        numRows, numCols = mI.shape[:2]
        hF, hA = plt.subplots(figsize = (int(np.ceil(numCols / dpi) + 1), int(np.ceil(numRows / dpi) + 1)))
    
    hA.imshow(mI, extent = [0, 1, 1, 0]) #<! "Normalized Image"
    hA.grid(False)

    PlotBBox(hA, boxLabel, vBox, boxScore)

    return hA

def PlotBBox( hA: plt.Axes, boxLabel: int, vBox: np.ndarray, boxScore: Optional[float] = None ) -> plt.Axes:
    # Assumes data in YOLO Format

    edgeColor = hA._get_lines.get_next_color()

    if boxScore is not None:
        labelText = f'Score: {boxScore:.0%}'
    else:
        labelText = '_' #<! https://stackoverflow.com/questions/24680981
    rectPatch = Rectangle((vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2)), vBox[2], vBox[3], linewidth = 2, edgecolor = edgeColor, facecolor = (0, 0, 0, 0), label = labelText) #<! Requires the alpha component in the face color
    hA.add_patch(rectPatch)
    hA.text(vBox[0] - (vBox[2] / 2), vBox[1] - (vBox[3] / 2), s = boxLabel, color = 'w', verticalalignment = 'bottom', bbox = {'color': edgeColor}, fontdict = {'size': 16})
    
    if boxScore is not None:
        hA.legend() #<! No need for legend unless there is a `Rectangle()`

    return hA