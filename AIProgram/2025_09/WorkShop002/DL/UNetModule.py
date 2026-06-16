

import torch
import torch.nn as nn

from enum import Enum, auto, unique

from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self: Self, numInChannels: int, numOutChannels: int, numMidChannels: Optional[int] = None, kernelSize = 3) -> None:
        super().__init__()
        if numMidChannels is None:
            numMidChannels = numOutChannels
            
        self.oDblConv = nn.Sequential(
            nn.Conv2d(numInChannels, numMidChannels, kernel_size = kernelSize, padding = 'same', bias = False),
            nn.BatchNorm2d(numMidChannels),
            nn.ReLU(inplace = True),
            nn.Conv2d(numMidChannels, numOutChannels, kernel_size = kernelSize, padding = 'same', bias = False),
            nn.BatchNorm2d(numOutChannels),
            nn.ReLU(inplace = True)
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        return self.oDblConv(x)


class Down(nn.Module):
    """Encoder Block - Downscaling with MaxPool then double Conv"""

    def __init__(self: Self, numInChannels: int, numOutChannels: int, kernelSize: int = 3) -> None:
        super().__init__()
        
        self.oMaxPoolConv = nn.Sequential(
            nn.MaxPool2d(2), #<! After skip connection, do downsample
            DoubleConv(numInChannels, numOutChannels, kernelSize = kernelSize)
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        return self.oMaxPoolConv(x)


class Up(nn.Module):
    """Decoder Block - Upscaling then double Conv"""

    def __init__(self: Self, numInChannels: int, numOutChannels: int, kernelSize: int = 3) -> None:
        super().__init__()

        self.oUpLyr   = nn.ConvTranspose2d(numInChannels, numOutChannels, kernel_size = 2, stride = 2, padding = 0)
        self.oConvLyr = DoubleConv(2 * numOutChannels, numOutChannels, kernelSize) #<! Royi: Assumption, the number of channels in filters are mult by 2

    def forward(self: Self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # One may use padding to support arbitrary dimensions
        
        x1 = self.oUpLyr(x1)
        x = torch.cat([x2, x1], dim = 1)
        
        return self.oConvLyr(x)


class OutConv(nn.Module):
    def __init__(self: Self, numInChannels: int, numOutChannels: int) -> None:
        super(OutConv, self).__init__()
        
        self.oConv = nn.Conv2d(numInChannels, numOutChannels, kernel_size = 1)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        
        return self.oConv(x)


class BuildUNet(nn.Module):
    def __init__(self: Self, numChannels: int, numClasses: int, filterSize: List[int], *, kernel_size: int = 3, oTtopLayer: Optional[Callable] = None) -> None:
        super(BuildUNet, self).__init__()
        # Assumption: filter_size[ii + 1] == 2 * filter_size[ii]
        # TODO: fix the above assumption by sending the actual size to the Up layer.
        # TODO: Check the code for small number of filters (1).
        # TODO: Works for input with even dimensions only
        
        self.numChannels = numChannels
        self.numClasses  = numClasses
        self.numFilters  = len(filterSize)
        self.filterSize  = filterSize

        self.oInConv = DoubleConv(numChannels, filterSize[0], kernelSize = kernel_size)
        print(f'In Layer: {filterSize[0]}')
        lDownLayer = []
        for ii in range(1, self.numFilters):
            lDownLayer.append(Down(filterSize[ii - 1], filterSize[ii], kernelSize = kernel_size))
            print(f'Down Layer: {filterSize[ii]}')
        
        lUpLayer = []
        for ii in range(1, self.numFilters):
            lUpLayer.append(Up(filterSize[-ii], filterSize[-(ii + 1)], kernelSize = kernel_size))
            print(f'Up Layer: {filterSize[-ii]}')

        # `nn.ModuleList` vs. `nn.Sequential` : https://stackoverflow.com/questions/47544051
        self.oDownLayer = nn.ModuleList(lDownLayer)
        self.oUpLayer   = nn.ModuleList(lUpLayer)

        self.oOutConv = OutConv(filterSize[0], numClasses)
        print(f'Out Layer: {filterSize[0]}')

        self.lBuff = [None] * self.numFilters #<! Buffer

        if oTtopLayer is None:
            self.oTtopLayer = nn.Identity()
        else:
            self.oTtopLayer = oTtopLayer

    def forward(self: Self, tX: torch.Tensor) -> torch.Tensor:
        
        self.lBuff[0] = self.oInConv(tX)
        for ii in range(1, self.numFilters):
            self.lBuff[ii] = self.oDownLayer[ii - 1](self.lBuff[ii - 1])
        
        tX = self.oUpLayer[0](self.lBuff[-1], self.lBuff[-2])
        for ii in range(1, self.numFilters - 1):
            tX = self.oUpLayer[ii](tX, self.lBuff[-(ii + 2)])

        tO = self.oOutConv(tX)
        tO = self.oTtopLayer(tO)
        
        return tO


