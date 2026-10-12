# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Fixel Algorithms - Generates MNIST Object Detection
# Generates the MNIST Object Detection dataset.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks
# - Requires installing PyAV: `micromamba install av -c conda-forge`.
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 11/10/2026 | Royi Avital | First version                                                                            |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning
from sklearn.datasets import fetch_openml

# Deep Learning

# Optimization

# Miscellaneous
from datetime import datetime
import json
import os
from platform import python_version
import random
import shutil

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_NAME       = 'FixelCourses'
DATA_FOLDER_NAME   = 'DataSets'
MODELS_FOLDER_NAME = 'Models'
BASE_FOLDER_PATH   = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH   = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)
MODELS_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, MODELS_FOLDER_NAME)

# %% Local Packages


# %% Auxiliary Functions

def GenMNISTObjDetDataset( dirPath: str, numSamples: int, imgSize: int, maxSizeDigit: int, minSizeDigit: int, maxNumDigits: int, seedNum: int ) -> bool:

    try:
        if numSamples < 1 or imgSize < 1 or maxNumDigits < 1:
            return False
        if not 1 <= minSizeDigit <= maxSizeDigit <= imgSize:
            return False

        mData, vLabels = fetch_openml('mnist_784', version = 1, as_frame = False, return_X_y = True)
        mDigits = np.asarray(mData, dtype = np.uint8).reshape(-1, 28, 28)
        vLabels = np.asarray(vLabels, dtype = np.int64)
        oRng = np.random.default_rng(seedNum)
        os.makedirs(dirPath, exist_ok = True)

        for sampleIdx in range(1, numSamples + 1):
            mImg = np.zeros((imgSize, imgSize), dtype = np.uint8)
            lLabels = []
            numDigits = int(oRng.integers(1, maxNumDigits + 1))

            for digitIdx in range(numDigits):
                for placementAttempt in range(100):
                    digitSampleIdx = int(oRng.integers(len(mDigits)))
                    mDigit = mDigits[digitSampleIdx]
                    vRows, vCols = np.nonzero(mDigit)
                    if len(vRows) == 0:
                        continue
                    mDigit = mDigit[vRows.min():vRows.max() + 1, vCols.min():vCols.max() + 1]
                    mDigit = np.pad(mDigit, 2, mode = 'constant')

                    digitSize = int(oRng.integers(minSizeDigit, maxSizeDigit + 1))
                    scaleFactor = digitSize / max(mDigit.shape)
                    digitHeight = max(1, round(mDigit.shape[0] * scaleFactor))
                    digitWidth = max(1, round(mDigit.shape[1] * scaleFactor))
                    mDigit = ski.transform.resize(mDigit, (digitHeight, digitWidth), order = 1,
                                                  mode = 'constant', cval = 0, preserve_range = True, anti_aliasing = True)
                    mDigit = np.rint(mDigit).astype(np.uint8)
                    vRows, vCols = np.nonzero(mDigit)
                    if len(vRows) == 0:
                        continue
                    mDigit = mDigit[vRows.min():vRows.max() + 1, vCols.min():vCols.max() + 1]
                    digitHeight, digitWidth = mDigit.shape

                    topRow = int(oRng.integers(imgSize - digitHeight + 1))
                    leftCol = int(oRng.integers(imgSize - digitWidth + 1))
                    mPatch = mImg[topRow:topRow + digitHeight, leftCol:leftCol + digitWidth]
                    if np.any((mPatch > 0) & (mDigit > 0)):
                        continue
                    np.maximum(mPatch, mDigit, out = mPatch)

                    centerCol = (leftCol + digitWidth / 2) / imgSize
                    centerRow = (topRow + digitHeight / 2) / imgSize
                    boxWidth = digitWidth / imgSize
                    boxHeight = digitHeight / imgSize
                    lLabels.append(f'{vLabels[digitSampleIdx]} {centerCol:.8f} {centerRow:.8f} {boxWidth:.8f} {boxHeight:.8f}\n')
                    break
                else:
                    break

            if not lLabels:
                return False
            sampleName = f'Img{sampleIdx:05d}'
            ski.io.imsave(os.path.join(dirPath, sampleName + '.png'), mImg, check_contrast = False)
            with open(os.path.join(dirPath, sampleName + '.txt'), 'w', encoding = 'ascii') as hFile:
                hFile.writelines(lLabels)

        dMetadata = {
            'date': datetime.now().strftime('%Y_%m_%d %H:%M'),
            'parameters': {
                'dirPath': os.fspath(dirPath),
                'numSamples': numSamples,
                'imgSize': imgSize,
                'maxSizeDigit': maxSizeDigit,
                'minSizeDigit': minSizeDigit,
                'maxNumDigits': maxNumDigits,
                'seedNum': seedNum,
            },
            'source': {'name': 'mnist_784', 'version': 1},
            'versions': {
                'python': python_version(),
                'numpy': np.__version__,
                'scikit-image': ski.__version__,
            },
        }
        with open(os.path.join(dirPath, 'GenerationParameters.json'), 'w', encoding = 'utf-8') as hFile:
            json.dump(dMetadata, hFile, indent = 4)
            hFile.write('\n')
    except Exception as oErr:
        print(f'MNIST object detection generation failed: {oErr}')
        return False

    return True


# %% Parameters

dirName  = 'MNISTObjDet'
fileName = 'MNISTObjDet.zip'

numSamples   = 15_000  #<! Number of samples to generate
imgSize      = 128     #<! [Pixel]
maxSizeDigit = 100     #<! [Pixel]
minSizeDigit = 16      #<! [Pixel]
maxNumDigits = 4       #<! Maximum number of digits per image

genZipFile = False

# %% Generate / Load Data

dirPath = os.path.join(DATA_FOLDER_PATH, dirName)


# %% Generate Data 

if not GenMNISTObjDetDataset(dirPath, numSamples, imgSize, maxSizeDigit, minSizeDigit, maxNumDigits, seedNum):
    raise RuntimeError('MNIST object detection dataset generation failed.')

# %% Display Samples
# Display samples to verify the generated dataset.
# Show a 3x3 grid of random samples from the generated dataset.

vSampleIdx = np.random.default_rng(seedNum).choice(numSamples, size = min(9, numSamples), replace = False) + 1
hFig, mAxes = plt.subplots(3, 3, figsize = (9, 9))
for hAxes, sampleIdx in zip(mAxes.flat, vSampleIdx):
    sampleName = f'Img{sampleIdx:05d}'
    hAxes.imshow(ski.io.imread(os.path.join(dirPath, sampleName + '.png')), cmap = 'gray', vmin = 0, vmax = 255)
    with open(os.path.join(dirPath, sampleName + '.txt'), 'r', encoding = 'ascii') as hFile:
        for labelLine in hFile:
            classId, centerCol, centerRow, boxWidth, boxHeight = map(float, labelLine.split())
            leftCol = (centerCol - boxWidth / 2) * imgSize
            topRow = (centerRow - boxHeight / 2) * imgSize
            rightCol = leftCol + boxWidth * imgSize
            bottomRow = topRow + boxHeight * imgSize
            boxColor = lMatPltLibclr[int(classId)]
            hAxes.plot([leftCol, rightCol, rightCol, leftCol, leftCol],
                       [topRow, topRow, bottomRow, bottomRow, topRow], color = boxColor)
            hAxes.text(leftCol, topRow, str(int(classId)), color = boxColor, va = 'bottom')
    hAxes.set_title(sampleName)
for hAxes in mAxes.flat:
    hAxes.axis('off')
hFig.tight_layout()

# %% Generate ZIP File

if genZipFile:
    zipPath = os.path.join(DATA_FOLDER_PATH, fileName)
    zipPath = shutil.make_archive(os.path.splitext(zipPath)[0], 'zip',
                                  root_dir = os.path.dirname(dirPath), base_dir = os.path.basename(dirPath))
    print(f'Created {zipPath}')
