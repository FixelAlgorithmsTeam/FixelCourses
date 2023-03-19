# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous
from enum import auto, Enum, unique
# import os
# import re
# import requests
# import string
# import time

# Typing
from typing import Dict, List, Tuple

# See https://docs.python.org/3/library/enum.html
# @unique
# class ModelType(Enum):
#     # Type of data in the CSV
#     MODEL_ADA      = auto()
#     MODEL_BABBAGE  = auto()
#     MODEL_CURIE    = auto()
#     MODEL_DAVINCI  = auto()

# Visualization

def PlotImages(mX: np.ndarray, vY: np.ndarray = None, numRows: int = 1, numCols: int = 1, tImgSize: Tuple[int, int] = (80, 80), numChannels: int = 1, randomChoice = True, hF = None) -> plt.Figure:

    numSamples  = mX.shape[0]

    numImg = min(numRows * numCols, numSamples)

    # tFigSize = (numRows * 3, numCols * 3)
    tFigSize = (numCols * 3, numRows * 3)

    if hF is None:
        hF, hA = plt.subplots(numRows, numCols, figsize = tFigSize)
    else:
        hA = hF.axis
    
    hA = np.atleast_1d(hA) #<! To support numImg = 1
    hA = hA.flat

    if randomChoice:
        vIdx = np.random.choice(numSamples, numImg, replace = False)
    else:
        vIdx = range(numImg)

    
    for kk in range(numImg):
        
        idx = vIdx[kk]
        mI  = np.reshape(mX[idx, :], tImgSize + (numChannels, ), order = 'F')
    
        if numChannels == 1:
            hA[kk].imshow(mI, cmap = 'gray')
        else:
            hA[kk].imshow(mI)
        hA[kk].tick_params(axis = 'both', left = False, top = False, right = False, bottom = False, labelleft = False, labeltop = False, labelright = False, labelbottom = False)
        # hA[kk].grid(False)
        labelStr = f', Label = {vY[idx]}' if vY is not None else ''
        hA[kk].set_title(f'Index = {idx}' + labelStr)
    
    return hF

def PlotLabelsHistogram(vY: np.ndarray, hA = None, lClass = None, xLabelRot: int = None) -> plt.Axes:

    if hA is None:
        hF, hA = plt.subplots(figsize = (8, 6))
    
    vLabels, vCounts = np.unique(vY, return_counts = True)

    hA.bar(vLabels, vCounts, width = 0.9, align = 'center')
    hA.set_title('Histogram of Classes / Labels')
    hA.set_xlabel('Class')
    hA.set_ylabel('Number of Samples')
    hA.set_xticks(vLabels)
    if lClass is not None:
        hA.set_xticklabels(lClass)
    
    if xLabelRot is not None:
        for xLabel in hA.get_xticklabels():
            xLabel.set_rotation(xLabelRot)

    return hA

def PlotConfusionMatrix(vY: np.ndarray, vYPred: np.ndarray, normMethod: str = None, hA: plt.Axes = None, lLabels: list = None, dScore: dict = None, titleStr: str = 'Confusion Matrix', xLabelRot: int = None, valFormat: str = None) -> Tuple[plt.Axes, np.ndarray]:

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