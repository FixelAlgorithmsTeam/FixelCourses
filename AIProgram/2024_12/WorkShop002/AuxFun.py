# %% Packages

# Python STD
from enum import auto, Enum, unique

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning
import torch
import torch.nn as nn

# Miscellaneous

# Visualization
import matplotlib.pyplot as plt


# %% Configuration


# %% Constants

# Matplotlib default color palette
L_MATPLOTLIB_COLOR = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# %% Auxiliary Classes

class SqueezeTrns(nn.Module):
    def __init__(self, dim: int = None) -> None:
        super().__init__()

        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return torch.squeeze(x, dim = self.dim)

class SubtractConst(nn.Module):
    def __init__(self, const: int = 0) -> None:
        super().__init__()

        self.const = const
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x - self.const


# %% Auxiliary Functions

def ModelToMask( tI: torch.Tensor ) -> np.ndarray:

    tI = torch.squeeze(tI, dim = 0)
    mM = torch.argmax(tI, dim = 0)
    mM = mM.cpu().numpy()

    return mM

def PlotMasks( mI: np.ndarray, mM: np.ndarray, *, mP: Optional[np.ndarray] = None ) -> plt.Figure:
    # mI -  Input Image
    # mM -  Input Mask
    # mP -  Predicted Mask (Optional)

    if mP is not None:
        numImg = 3
    else:
        numImg = 2
    
    hF, vHa = plt.subplots(nrows = 1, ncols = numImg, figsize = (5 * numImg, 5))

    vHa = vHa.flat
    hA = vHa[0]
    hA.imshow(mI)
    hA.axis('off')
    hA.set_title('Input Image')

    hA = vHa[1]
    hA.imshow(mM, interpolation = 'nearest')
    hA.axis('off')
    hA.set_title('Input Mask')

    if (numImg == 3):
        hA = vHa[2]
        hA.imshow(mP, interpolation = 'nearest')
        hA.axis('off')
        hA.set_title('Predicted Mask')
    
    return hF


def UnNormalizeImg( tI: torch.Tensor, vMean: List[float], vStd: List[float] ) -> torch.Tensor:
    """
    UnNormalizes an image tensor by applying the inverse of the normalization transformation.  
    The normalization is defined by the mean and standard deviation per channel.
    Input:
        tI          - PyTorch `Tensor` of the image to unnormalize.
        vMean       - List of mean values per channel.
        vStd        - List of standard deviation values per channel.
    Output:
        tI          - Unnormalized image tensor.
    Remarks:
      - The tensor `tI` is expected to have shape `(C, H, W)`, where `C` is the number of channels (3 for RGB).
    """
    tI = tI * torch.tensor(vStd).view(3, 1, 1) + torch.tensor(vMean).view(3, 1, 1)
    
    return tI

def DataTensorToImageMask( tI: torch.Tensor, tM: torch.Tensor, vMean: List[float], vStd: List[float] ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a data tensor to an image and its corresponding mask.
    Input:
        tI          - PyTorch `Tensor` of the image.
        tM          - PyTorch `Tensor` of the mask.
        vMean       - List of mean values per channel.
        vStd        - List of standard deviation values per channel.
    Output:
        mI          - Numpy array of the unnormalized image.
        mM          - Numpy array of the mask.
    Remarks:
      - The tensor `tI` is expected to have shape `(1, C, H, W)` or `(C, H, W)`.
      - The output images `mI` and `mM` will have shapes `(H, W, C)` and `(H, W)`.
    """
    tI = torch.squeeze(tI, dim = 0) #<! Remove batch dimension (If exist)
    tI = UnNormalizeImg(tI, vMean, vStd) #<! Reverse the normalization step
    mI = np.permute_dims(tI.cpu().numpy(), (1, 2, 0)) #<! (C, H, W) -> (H, W, C)
    mM = tM.cpu().numpy() #<! Convert ot NumPy array

    return mI, mM

