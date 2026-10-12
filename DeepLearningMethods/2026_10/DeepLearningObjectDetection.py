
# Python STD
from enum import auto, Enum, unique
import math
import os
import time

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
from torchvision.transforms.v2 import functional as TorchVisionTrnsF

# Image Processing / Computer Vision
import imageio.v3 as iio
import skimage as ski

# Optimization

# Auxiliary

# Visualization

# Miscellaneous

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
        
        self.tX         = tX #<! (numSamples, C, H, W)
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
        
        self._tX = tX
        self._lY = lY
        self._lB = lB
        self._numSamples = tX.shape[0]
        self._hDataTrans = hDataTrans

    def __len__( self: Self ) -> int:
        
        return self._numSamples

    def __getitem__( self: Self, idx: int ) -> Tuple[NDArray | Tensor, Tuple[NDArray | Tensor, NDArray | Tensor]]:
        
        tXi = self._tX[idx] #<! Image
        vYi = self._lY[idx] #<! Labels
        mBi = self._lB[idx] #<! Bounding Boxes

        tXi = tXi.astype(np.float32)
        vYi = vYi.astype(np.int64)
        mBi = mBi.astype(np.float32)

        if self._hDataTrans is not None:
            tXi, vYi, mBi = self._hDataTrans(tXi, vYi, mBi)
        
        return tXi, (vYi, mBi)

    def GetFeatures( self: Self ) -> NDArray:

        return self._tX

    def GetLabels( self: Self ) -> List[NDArray]:

        return self._lY

    def GetBoundingBoxes( self: Self ) -> List[NDArray]:

        return self._lB

    def GetClasses( self: Self ) -> Set[int]:

        vY = np.concatenate(self._lY, axis = 0)

        return set(vY.tolist())

    def SetTransform( self: Self, hDataTrans: Callable ) -> None:

        self._hDataTrans = hDataTrans

class YoloDataset( Dataset ):
    """Load YOLO labels as pixel XYXY boxes for torchvision v2 transforms.

    Cache all labels and normalized YOLO boxes in memory, assuming no more than
    100,000 objects in total across the dataset. Images are decoded on demand
    with torchvision.
    Use the `imgMode` parameter to specify the desired image read mode when decoding images with `torchvision.io.decode_image`.
    Use `torchvision.io.ImageReadMode.RGB` to read images in RGB mode and `torchvision.io.ImageReadMode.GRAY` to read images in grayscale mode.
    """

    def __init__( self, dirPath: str, /, *, hDataTrans: Optional[Callable] = None, lImgExt: List[str] = ['.jpg', '.png'], imgMode: torchvision.io.ImageReadMode = torchvision.io.ImageReadMode.RGB ) -> None:

        lImgName    = sorted(
            imgName for imgName in os.listdir(dirPath)
            if os.path.splitext(imgName)[1].lower() in lImgExt
            and os.path.isfile(os.path.join(dirPath, imgName))
        )

        lY: List[Tensor] = []
        lB: List[Tensor] = []
        dL: Set[int]     = set() #<! Set of class labels
        for imgName in lImgName:
            labelPath = os.path.join(dirPath, os.path.splitext(imgName)[0] + '.txt')
            with open(labelPath, 'r', encoding = 'utf-8') as hFile:
                lLines = [line for line in hFile if line.strip()]
            mAnn = np.loadtxt(lLines, dtype = np.float32, ndmin = 2) if lLines else np.empty((0, 5), dtype = np.float32)
            if mAnn.shape[1] != 5:
                raise ValueError(f'Expected 5 columns in YOLO label file: {labelPath}')

            lY.append(torch.from_numpy(mAnn[:, 0]).to(torch.int64))
            lB.append(torch.from_numpy(mAnn[:, 1:]).clone())
            dL.update(lY[-1].tolist())
        
        self._dirPath     = dirPath
        self._hDataTrans  = hDataTrans
        self._lImgExt     = lImgExt
        self._imgMode     = imgMode
        self._lImgName    = lImgName
        self._lY          = lY #<! List of `torch.Tensor` vectors (Class labels)
        self._lB          = lB #<! List of `torch.Tensor` matrices (Bounding boxes)
        self._numSamples  = len(self._lImgName)
        self._dL          = dL #<! Sorted unique class labels of the dataset
        self._numClasses  = len(dL)

    def __len__( self: Self ) -> int:
        
        return self._numSamples

    def __getitem__( self: Self, idx: int ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # The output bounding box is in `XYXY` pixel format to optimize calculation of IoU and other geometric operations.

        imgName   = self._lImgName[idx]
        imgPath   = os.path.join(self._dirPath, imgName)

        tXi = torchvision.io.decode_image(imgPath, mode = self._imgMode)
        tXi = torchvision.tv_tensors.Image(tXi)
        tXi = TorchVisionTrnsF.to_dtype(tXi, torch.float32, scale = True)
        numRows, numCols = tXi.shape[-2:]

        vYi = self._lY[idx].clone()
        mBi = self._lB[idx].clone()
        mBi *= mBi.new_tensor([numCols, numRows, numCols, numRows])
        mBi = torchvision.ops.box_convert(mBi, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        mBi = torchvision.tv_tensors.BoundingBoxes(mBi, format = 'XYXY', canvas_size = (numRows, numCols))

        if self._hDataTrans is not None:
            tXi, dTarget = self._hDataTrans(tXi, {'labels': vYi, 'boxes': mBi}) #<! Using a dictionary for compatibility with `torchvision.v2.SanitizeBoundingBoxes`
            vYi, mBi = dTarget['labels'], dTarget['boxes']
        
        return tXi, (vYi, mBi)

    def SetTransform( self: Self, hDataTrans: Callable ) -> None:
        
        self._hDataTrans = hDataTrans
    
    def GetTargets( self: Self ) -> Tuple[List[Tensor], List[Tensor]]:
            
        return self._lY, self._lB

    def GetLabelsSet( self: Self ) -> Set[int]:
                
        return self._dL.copy()

    def GetNumClasses( self: Self ) -> int:
        
        return self._numClasses

# Functions

def CollateObjectDetection( lBatch: List[Tuple[NDArray, Tuple[NDArray, NDArray]]] ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Collate function for object detection with variable number of boxes per image.

        Expects each dataset item to be:
            (tXi, (vYi, mBi))
        where:
            tXi : Image like array / tensor (same shape across the batch)
            vYi : Labels with shape (Di,) (variable Di)
            mBi : Bounding boxes with shape (Di, 4) (variable Di)

        Returns:
            tX : Tensor of shape (N, ...) stacked over the batch
            lT : List (len N) of dicts: {'Labels': Tensor[Di], 'Boxes': Tensor[Di, 4]}
        """

        lX: List[Tensor] = []
        lT: List[Dict[str, Tensor]] = []

        for x, (y, b) in lBatch:
            tXi = torch.tensor(x)
            vYi = torch.tensor(y)
            mBi = torch.tensor(b)

            # Common dtypes for detection targets
            tXi = tXi.to(torch.float32)
            if vYi.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
                vYi = vYi.to(torch.int64)
            mBi = mBi.to(torch.float32)

            lX.append(tXi)
            lT.append({'Labels': vYi, 'Boxes': mBi})

        # Images must be stackable (same shape)
        tXb = torch.stack(lX, dim = 0)

        return tXb, lT

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

