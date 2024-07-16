

import torch
import torch.nn as nn

from enum import Enum, auto, unique

from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None, kernel_size = 3):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = kernel_size, padding = 'same', bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = kernel_size, padding = 'same', bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Encoder Block - Downscaling with MaxPool then double Conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), #<! After skip connection, do downsample
            DoubleConv(in_channels, out_channels, kernel_size = kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Decoder Block - Upscaling then double Conv"""

    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()

        self.up     = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2, padding = 0)
        self.conv   = DoubleConv(2 * out_channels, out_channels, kernel_size) #<! Royi: Assumption, the number of channels in filters are mult by 2

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)


class BuildUNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, filter_size: List, *, kernel_size: int = 3, top_layer: Optional[Callable] = None):
        super(BuildUNet, self).__init__()
        # Assumption: filter_size[ii + 1] == 2 * filter_size[ii]
        # TODO: fix the above assumption by sending the actual size to the Up layer.
        # TODO: Check the code for small number of filters (1).
        # TODO: Works for input with even dimensions only
        
        self.n_channels     = n_channels
        self.n_classes      = n_classes
        self.num_filters    = len(filter_size)
        self.filter_size    = filter_size

        self.inc = DoubleConv(n_channels, filter_size[0], kernel_size = kernel_size)
        print(f'In Layer: {filter_size[0]}')
        down_layer = []
        for ii in range(1, self.num_filters):
            down_layer.append(Down(filter_size[ii - 1], filter_size[ii], kernel_size = kernel_size))
            print(f'Down Layer: {filter_size[ii]}')
        
        up_layer = []
        for ii in range(1, self.num_filters):
            up_layer.append(Up(filter_size[-ii], filter_size[-(ii + 1)], kernel_size = kernel_size))
            print(f'Up Layer: {filter_size[-ii]}')

        # `nn.ModuleList` vs. `nn.Sequential` : https://stackoverflow.com/questions/47544051
        self.down_layer = nn.ModuleList(down_layer)
        self.up_layer   = nn.ModuleList(up_layer)

        self.outc = OutConv(filter_size[0], n_classes)
        print(f'Out Layer: {filter_size[0]}')

        self.buff = [None] * self.num_filters

        if top_layer is None:
            self.top_layer = nn.Identity()
        else:
            self.top_layer = top_layer

    def forward(self, x) -> torch.Tensor:
        
        self.buff[0] = self.inc(x)
        for ii in range(1, self.num_filters):
            self.buff[ii] = self.down_layer[ii - 1](self.buff[ii - 1])
        
        x = self.up_layer[0](self.buff[-1], self.buff[-2])
        for ii in range(1, self.num_filters - 1):
            x = self.up_layer[ii](x, self.buff[-(ii + 2)])

        out_layer = self.outc(x)
        out_layer = self.top_layer(out_layer)
        
        return out_layer


