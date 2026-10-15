
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
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union
from numpy.typing import ArrayLike, NDArray

# Course Packages

# Constants
FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2

def PlotLabelsHistogram( vY: NDArray, hA: Optional[plt.Axes] = None, lClass: Optional[List] = None, xLabelRot: Optional[int] = None ) -> plt.Axes:

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

def PlotConfusionMatrix(vY: NDArray, vYPred: NDArray, *, normMethod: str = None, hA: Optional[plt.Axes] = None, 
                        lLabels: Optional[List] = None, dScore: Optional[Dict] = None, titleStr: str = 'Confusion Matrix', 
                        xLabelRot: Optional[int] = None, valFormat: Optional[str] = None) -> Tuple[plt.Axes, NDArray]:

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
