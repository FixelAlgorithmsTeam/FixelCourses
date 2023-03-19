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

# Miscellaneous
from typing import Any, Callable, List, Optional, Union, Tuple

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