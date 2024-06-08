
# Python STD
from enum import auto, Enum, unique
# import math
import shutil

# Data
import numpy as np
# import pandas as pd
# import scipy as sp

# Machine Learning

# Image Processing / Computer Vision
import skimage as ski

# Optimization

# Auxiliary

# Visualization

# Miscellaneous
import gdown
import gzip
import os
import urllib.request

# Typing
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

# See https://docs.python.org/3/library/enum.html
@unique
class DiffMode(Enum):
    # Type of data in the CSV
    BACKWARD    = auto()
    CENTRAL     = auto()
    FORWARD     = auto()
    COMPLEX     = auto()

@unique
class BBoxFormat(Enum):
    # Bounding Box Format, See https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation
    COCO        = auto()
    PASCAL_VOC  = auto()
    YOLO        = auto()

# Constants
L_ARCHIVE_EXT = ['.zip', '.tar.bz2', '.bz2', '.tbz2', '.tar.gz', '.gz', '.tgz', '.tar', '.tar.xz', '.xz', '.txz']

def DownloadGDriveZip( fileId: str, lFileCont: List[str] ) -> None:

    for fileName in lFileCont:
        if os.path.isfile(fileName):
            os.remove(fileName)
    
    fileNameExt = gdown.download(id = fileId)
    fileName, fileExt = os.path.splitext(fileNameExt)
    if fileExt in L_ARCHIVE_EXT:
        # Might not work with `tar` files (Might require unpacking twice)
        shutil.unpack_archive(fileNameExt)
        os.remove(fileNameExt)

def DownloadDecompressGzip( fileUrl: str, fileName: str) -> None:
    # Based on https://stackoverflow.com/a/61195974

    # Read the file inside the .gz archive located at url
    with urllib.request.urlopen(fileUrl) as response:
        with gzip.GzipFile(fileobj = response) as uncompressed:
            file_content = uncompressed.read()
        # write to file in binary mode 'wb'
        with open(fileName, 'wb') as f:
            f.write(file_content)
            f.close()
        return

def DownloadUrl( fileUrl: str, fileName: str ) -> str:
    
    if not os.path.exists(fileName):
        urllib.request.urlretrieve(fileUrl, fileName)

    return fileName

def ConvertMnistDataDf( imgFilePath: str, labelFilePath: str ) -> Tuple[np.ndarray, np.ndarray]:
    numPx = 28 * 28
    # Merge of https://pjreddie.com/projects/mnist-in-csv/ and https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py
    f = open(imgFilePath, "rb")
    l = open(labelFilePath, "rb")

    lCol = [f'Px {ii:04}' for ii in range (numPx)]
    lCol.append('Label')

    vY = np.frombuffer(l.read(), np.uint8, offset = 8)
    mX = np.frombuffer(f.read(), np.uint8, offset = 16)
    # mX = np.reshape(mX, (numPx, len(vY))).T
    mX = np.reshape(mX, (len(vY), numPx))

    f.close()
    l.close()

    return mX, vY


def ConvertBBoxFormat( vBox: np.ndarray, tuImgSize: Tuple[int, int], boxFormatIn: BBoxFormat, boxFormatOut: BBoxFormat ) -> np.ndarray:
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

def GenLabeldEllipseImg( tuImgSize: Tuple[int, int], numObj: int, *, boxFormat: BBoxFormat = BBoxFormat.YOLO ) -> Tuple[np.ndarray, np.ndarray]:
    # Image Size in Rows x Cols

    mI  = np.zeros(shape = (*tuImgSize, 3)) #<! RGB Image
    vY  = np.zeros(shape = numObj, dtype = np.int_)
    mBB = np.zeros(shape = (numObj, 4)) #<! [x, y, width, height]

    for ii in range(numObj):
        cIdx    = np.random.randint(3) #<! R, G, B -> [0, 1, 2]
        rotDeg  = np.pi * np.random.rand()
        centRow = np.random.randint(low = int(np.ceil(0.1 * tuImgSize[0])), high = int(np.ceil(0.9 * tuImgSize[0])))
        centCol = np.random.randint(low = int(np.ceil(0.1 * tuImgSize[1])), high = int(np.ceil(0.9 * tuImgSize[1])))
        majAxis = (tuImgSize[0] / 16) + ((tuImgSize[0] / 4) * np.random.rand()) #<! Major Axis
        minAxis = (tuImgSize[1] / 16) + ((tuImgSize[1] / 4) * np.random.rand()) #<! Minor Axis

        # Generate the Ellipse
        vR, vC = ski.draw.ellipse(centRow, centCol, majAxis, minAxis, shape = tuImgSize, rotation = rotDeg)

        mI[vR, vC, cIdx] = 1.0

        xLeft   = np.min(vC)
        xRight  = np.max(vC)
        yTop    = np.min(vR)
        yBottom = np.max(vR)

        vY[ii]     = cIdx       #<! Label
        mBB[ii, 0] = xLeft      #<! x Min
        mBB[ii, 1] = yTop       #<! y Min
        mBB[ii, 2] = xRight     #<! x Max
        mBB[ii, 3] = yBottom    #<! y Max

        if (boxFormat != BBoxFormat.PASCAL_VOC):
            mBB[ii] = ConvertBBoxFormat(mBB[ii], tuImgSize, BBoxFormat.PASCAL_VOC, boxFormat)

    return mI, vY, mBB