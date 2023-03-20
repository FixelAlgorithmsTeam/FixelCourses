# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
import skorch
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms

# Computer Vision
from skimage.io import imread

# Miscellaneous
import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from AuxFun import *

DEFAULT_IMG_FORMAT = ['.png']

# Model

class NnImageCls(nn.Module):
    def __init__(self, numClass = 2):
        super().__init__()
        modelResNet = models.resnet18(weights = None, progress = False)
        # Replace last level fully connected layer with 2 layers
        numFeatIn = modelResNet.fc.in_features
        modelResNet.fc = nn.Linear(numFeatIn, numClass) #<! The number of output features match number of classes
        self.modelCls = modelResNet
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelCls(x)


# Data Loaders

class DataFrameImgLoader(Dataset):
    def __init__(self, dfX: pd.DataFrame, dsY: pd.Series, tuImgSize: Tuple[int, int, int], imgTransform: Callable = None):
        
        self.mX = dfX.to_numpy()
        self.vY = dsY.to_numpy()
        
        self.tuImgSize = tuImgSize

    def __len__(self) -> int:
        return self.mX.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        mX = self.mX
        vY = self.Vy

        tImg     = np.reshape(mX[idx], self.tuImgSize, order = 'F') #<! Data from Pandas is column (Fortran) contiguous
        imgClass = vY[idx] #<! PyTorch expects labels in Int64 / Long format

        return tImg, imgClass

class NumPyArrayImgLoader(Dataset):
    def __init__(self, mX: np.ndarray, vY: np.ndarray, tuImgSize: Tuple[int, int, int], imgTransform: Callable = None):
        
        self.mX = mX
        self.vY = vY
        
        self.tuImgSize = tuImgSize
        self.imgTransform = imgTransform

    def __len__(self) -> int:
        return self.mX.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        mX = self.mX
        vY = self.Vy

        tImg     = np.reshape(mX[idx], self.tuImgSize, order = 'F') #<! Data from Pandas is column (Fortran) contiguous
        if vY is not None:
            imgClass = vY[idx] #<! PyTorch expects labels in Int64 / Long format
        else:
            imgClass = -1 #<! To support the case of .predict()

        return self.__transform__(tImg), imgClass
    
    def __transform__(self, tI: np.ndarray) -> np.ndarray:
        
        if self.imgTransform is not None:
            return self.imgTransform(tI)
        else:
            return transforms.ToTensor(tI) #<! Assuming data is UInt8 or properly scaled

class ShipsImgLoader(Dataset):
    def __init__(self, mX: np.ndarray, vY: np.ndarray):
        
        self.mX = mX
        self.vY = vY
        
        self.tuImgSize = (80, 80, 3)
        self.imgTransform = None

    def __len__(self) -> int:
        return self.mX.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        mX = self.mX
        vY = self.vY

        tImg     = np.reshape(mX[idx], self.tuImgSize, order = 'F') #<! Data from Pandas is column (Fortran) contiguous
        if vY is not None:
            imgClass = vY[idx] #<! PyTorch expects labels in Int64 / Long format
        else:
            imgClass = -1 #<! To support the case of .predict()

        return self.__transform__(tImg), imgClass
    
    def __transform__(self, tI: np.ndarray) -> np.ndarray:
        
        if self.imgTransform is not None:
            return self.imgTransform(tI)
        else:
            return transforms.functional.to_tensor(tI) #<! Assuming data is UInt8 or properly scaled


class PascalVocLoader(Dataset):
    def __init__(self, imgFolderPath: str, annFolderPath: str, dataTransform: Callable = None, lImgFormat: List = DEFAULT_IMG_FORMAT):
        
        lImgFolder = os.listdir(imgFolderPath)
        lImg = []
        for imgFileName in lImgFolder:
            fileFullPath = os.path.join(imgFolderPath, imgFileName)
            if os.path.isfile(fileFullPath):
                fileName, fileExt = os.path.splitext(imgFileName)
                if fileExt in lImgFormat:
                    mI = imread(fileFullPath)
                    lImg.append(mI[:, :, :3]) #<! Remove Alpha channel (Some images have it)

        lAnnFolder = os.listdir(annFolderPath)
        lBox = []

        for imgFileName in lAnnFolder:
            fileFullPath = os.path.join(annFolderPath, imgFileName)
            if os.path.isfile(fileFullPath):
                fileName, fileExt = os.path.splitext(imgFileName)
                if fileExt == '.xml':
                    lBox.append(ExtractBoxXml(fileFullPath))
        
        self.lImg           = lImg
        self.lBox           = lBox
        self.dataTransform  = dataTransform

    def __len__(self) -> int:
        return len(self.lImg)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        mI      = self.lImg[idx]
        lBox    = self.lBox[idx]

        return self.__transform__(mI, lBox)
    
    def __transform__(self, mI: np.ndarray, lBox: List) -> Tuple[torch.Tensor, List]:
        
        if self.dataTransform is not None:
            return self.dataTransform(mI, lBox)
        else:
            return transforms.functional.to_tensor(mI), torch.tensor(lBox)


class TorchObjDetectionCollateFn():
    # Supports a single class of bounding boxes
    def __init__(self, classId: int = 1):
        
        self.classId = classId
    
    def __call__(self, batchData: List[Tuple]) -> Tuple[List[torch.Tensor], List[Dict]]:
        
        lTarget = []
        lImg    = []
        for tI, tBox in batchData:
            dTarget = {}
            dTarget['boxes']  = tBox
            dTarget['labels'] = torch.tensor(self.classId).repeat(tBox.shape[0])
            lTarget.append(dTarget)
            lImg.append(tI)

        return lImg, lTarget