
# Python STD
from enum import auto, Enum, unique
import math

# Data
import numpy as np
# import scipy as sp
# import pandas as pd

# Machine Learning

# Deep Learning
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

# Image Processing / Computer Vision
import skimage as ski

# Optimization

# Auxiliary

# Visualization

# Miscellaneous
import time

# Course Packages
from DeepLearningBlocks import NNMode


# Typing
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union
from numpy.typing import NDArray
from torch import Tensor

# Auxiliary
@unique
class BBoxFormat(Enum):
    # Bounding Box Format, See https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation
    COCO        = auto()
    PASCAL_VOC  = auto()
    YOLO        = auto()

# Classes

class ObjectLocalizationDataset( Dataset ):
    def __init__( self, tX: NDArray, vY: NDArray, mB: NDArray, singleY: bool = True ) -> None:

        if (tX.shape[0] != vY.shape[0]):
            raise ValueError(f'The number of samples in `tX` and `vY` does not match!')
        if (tX.shape[0] != mB.shape[0]):
            raise ValueError(f'The number of samples in `tX` and `mB` does not match!')
        
        self.tX         = tX #<! (numSamples, H, W, C)
        self.vY         = vY #<! (numSamples, )
        self.mB         = mB #<! (numSamples, 4)
        self.singleY    = singleY #<! Return label and box, or a single vector
        self.numSamples = tX.shape[0]

    def __len__( self: Self ) -> int:
        
        return self.numSamples

    def __getitem__( self: Self, idx: int ) -> Union[Tuple[NDArray, int, NDArray], Tuple[NDArray, NDArray]]:
        
        tXi   = self.tX[idx] #<! Image
        valYi = self.vY[idx] #<! Label
        vBi   = self.mB[idx] #<! Bounding Box

        tXi   = tXi.astype(np.float32)
        vBi   = vBi.astype(np.float32)

        if self.singleY:
            valYi = valYi.astype(np.float32)
            return tXi, np.r_[valYi, vBi]
        else:
            return tXi, valYi, vBi

class ObjectDetectionDataset( Dataset ):
    def __init__( self, tX: NDArray, lY: List[NDArray], lB: List[NDArray], hDataTrans: Optional[Callable] = None ) -> None:

        if (tX.shape[0] != len(lY)):
            raise ValueError(f'The number of samples in `tX` and `lY` does not match!')
        if (tX.shape[0] != len(lB)):
            raise ValueError(f'The number of samples in `tX` and `lB` does not match!')
        
        self.tX = tX
        self.lY = lY
        self.lB = lB
        self.numSamples = tX.shape[0]
        self.hDataTrans = hDataTrans

    def __len__( self: Self ) -> int:
        
        return self.numSamples

    def __getitem__( self: Self, idx: int ) -> Union[Tuple[NDArray, int, NDArray], Tuple[NDArray, NDArray]]:
        
        tXi = self.tX[idx] #<! Image
        vYi = self.lY[idx] #<! Labels
        mBi = self.lB[idx] #<! Bounding Boxes

        tXi = tXi.astype(np.float32)
        vYi = vYi.astype(np.float32)
        mBi = mBi.astype(np.float32)

        mYi = np.c_[vYi, mBi]

        if self.hDataTrans is not None:
            tXi, mYi = self.hDataTrans(tXi, mYi)
        
        return tXi, mYi

# Functions

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

def GenLabeldEllipseImg( tuImgSize: Tuple[int, int], numObj: int, *, boxFormat: BBoxFormat = BBoxFormat.YOLO ) -> Tuple[NDArray, NDArray]:
    # Image Size in Rows x Cols
    # 3 Classes (R, G, B) -> 0, 1, 2

    mI  = np.zeros(shape = (*tuImgSize, 3)) #<! RGB Image
    vY  = np.zeros(shape = numObj, dtype = np.int_)
    mBB = np.zeros(shape = (numObj, 4)) #<! [x1, y1, x2, y2]

    for ii in range(numObj):
        cIdx    = np.random.randint(3)     #<! R, G, B -> [0, 1, 2]
        rotDeg  = np.pi * np.random.rand() #<! [0, π]
        centRow = np.random.randint(low = int(np.ceil(0.1 * tuImgSize[0])), high = int(np.ceil(0.9 * tuImgSize[0])))
        centCol = np.random.randint(low = int(np.ceil(0.1 * tuImgSize[1])), high = int(np.ceil(0.9 * tuImgSize[1])))
        majAxis = (tuImgSize[0] / 16) + ((tuImgSize[0] / 4) * np.random.rand()) #<! Major Axis
        minAxis = (tuImgSize[1] / 16) + ((tuImgSize[1] / 4) * np.random.rand()) #<! Minor Axis

        # Generate the Ellipse
        vR, vC = ski.draw.ellipse(centRow, centCol, majAxis, minAxis, shape = tuImgSize, rotation = rotDeg)

        mI[vR, vC, cIdx] = 1.0 #<! Class of the ellipse

        # Bounding Box
        xLeft   = np.min(vC)
        xRight  = np.max(vC)
        yTop    = np.min(vR)
        yBottom = np.max(vR)

        # PASCAL VOC format
        vY[ii]     = cIdx    #<! Label
        mBB[ii, 0] = xLeft   #<! x Min
        mBB[ii, 1] = yTop    #<! y Min
        mBB[ii, 2] = xRight  #<! x Max
        mBB[ii, 3] = yBottom #<! y Max

        if (boxFormat != BBoxFormat.PASCAL_VOC):
            mBB[ii] = ConvertBBoxFormat(mBB[ii], tuImgSize, BBoxFormat.PASCAL_VOC, boxFormat)

    return mI, vY, mBB

