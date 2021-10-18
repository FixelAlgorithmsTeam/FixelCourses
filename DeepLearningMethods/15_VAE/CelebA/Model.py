# %%
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

import torch
import torch.nn               as nn
import torch.nn.functional    as F
import torchvision.transforms as transforms
import torchinfo

# %%
class ConvBlock(nn.Module):
    def __init__(self, cIn, cOut, kernelSize, stride=1, bActivation=True):
        super(ConvBlock, self).__init__()

        padding = kernelSize // 2
        oConv   = nn.Conv2d(cIn, cOut, kernel_size=kernelSize, padding=padding, stride=stride, bias=not bActivation)
        if bActivation == True:
            self.oBlock = nn.Sequential(oConv, nn.BatchNorm2d(cOut), nn.LeakyReLU(0.1))
        else:
            self.oBlock = nn.Sequential(oConv)

    def forward(self, mX):
        return self.oBlock(mX)
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
class ResBlock(nn.Module):
    def __init__(self, cIn):
        super(ResBlock, self).__init__()
        
        self.oBlock = nn.Sequential(
            ConvBlock(cIn,    cIn//2, kernelSize=1),
            ConvBlock(cIn//2, cIn,    kernelSize=3)
        )

    def forward(self, mX):
        return mX + self.oBlock(mX)
#-------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------#
class Encoder(nn.Module):
    def __init__(self, D):
        super(Encoder, self).__init__()
        self.oEncoder = nn.Sequential(
            ConvBlock(3,   16,  3),
            ResBlock (16),
            ConvBlock(16,  32,  3, 2),
            ResBlock (32),
            ResBlock (32),
            ConvBlock(32,  64, 3, 2),
            ResBlock (64),
            ResBlock (64),
            ConvBlock(64,  128, 3, 2),
            ResBlock (128),
            ResBlock (128),
            ConvBlock(128, 256, 3, 2),
            ResBlock (256),
            ResBlock (256),
            ConvBlock(256, 512, 3, 2),
            ResBlock (512),
            ResBlock (512),
            nn.Conv2d(512, 2*D, 3, padding=1, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, mX):
        mOut      = self.oEncoder(mX)
        mμ, mLogΣ = mOut.chunk(2, dim=1)
            
        return mμ, mLogΣ
# %%
class Decoder(nn.Module):
    def __init__(self, D):
        super(Decoder, self).__init__()

        self.oDecoder = nn.Sequential(
            nn.Upsample(scale_factor=2), ConvBlock(D, 512, 1),
            nn.Upsample(scale_factor=2), ResBlock (512),
                                         ResBlock (512),
            nn.Upsample(scale_factor=2), ConvBlock(512, 256, 3),
            nn.Upsample(scale_factor=2), ResBlock (256),
                                         ResBlock (256),
                                         ConvBlock(256, 128, 3),
            nn.Upsample(scale_factor=2), ResBlock (128),
                                         ResBlock (128),
                                         ConvBlock(128, 64, 3),
            nn.Upsample(scale_factor=2), ResBlock (64),
                                         ResBlock (64),
                                         nn.Conv2d(64, 3, 1),
            nn.Tanh()
    )
        
    def forward(self, mX):
        return self.oDecoder(mX[:,:,None,None])
# %%
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        D             = 128
        self.D        = D
        self.oEncoder = Encoder(D)
        self.oDecoder = Decoder(D)

    def forward(self, mX):
        #-- Encode:
        mμ, mLogΣ = self.oEncoder(mX)

        #-- Sample:
        if self.training == True:
            mϵ = torch.randn_like(mμ)
            vσ = torch.exp(.5 * mLogΣ)
            mZ = vσ * mϵ + mμ
        else:
            mZ = mμ

        #-- Decode:
        mHatX = self.oDecoder(mZ)

        return mHatX, mμ, mLogΣ

if __name__ == '__main__':
    torchinfo.summary(VAE(), (128, 3, 64, 64))