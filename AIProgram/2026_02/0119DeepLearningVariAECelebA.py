# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Generative Models - Variational Auto Encoder - Celeb A
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 10/08/2026 | Royi Avital | First version                                                      |

# %% Packages

# General Tools
import numpy as np

# Deep Learning
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2 as TorchVisionTrns
from torchmetrics.functional import r2_score

# Computer Vision
import imageio

# Miscellaneous
import os
import random

# Typing
from typing import Callable, Literal, Optional, Self, Tuple, Union
from numpy.typing import NDArray
from torch import Tensor

# %% Configuration

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# Improve performance by benchmarking
torch.backends.cudnn.benchmark = True

# %% Constants

PROJECT_NAME       = 'FixelCourses'
DATA_FOLDER_NAME   = 'DataSets'
MODELS_FOLDER_NAME = 'Models'
BASE_FOLDER_PATH   = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH   = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)
MODELS_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, MODELS_FOLDER_NAME)

# %% Course Packages

from DeepLearningPyTorch import TrainModelDataset

# %% Auxiliary Functions

class CelebADataset(torch.utils.data.Dataset):
    """
    CelebA Dataset from a folder of images and a labels file.
    Supports image classification and self-supervised learning tasks.
    """

    def __init__( self: Self, datasetPath: str, *, clsFileName: str = 'Classes.txt', hImgTrns: Optional[Callable] = None, hFeatTrns: Optional[Callable] = None, hTgtTrns: Optional[Callable] = None ) -> None:

        lFiles = sorted([fileName for fileName in os.listdir(datasetPath) if fileName.endswith('.jpg')])
        vLbl   = np.loadtxt(os.path.join(datasetPath, clsFileName), dtype = np.int64)

        tI        = imageio.v3.imread(os.path.join(datasetPath, lFiles[0]))
        tuImgSize = (tI.shape[0], tI.shape[1])

        self._datasetPath = datasetPath
        self._numSamples  = len(lFiles)
        self._lFiles      = lFiles
        self._vLbl        = vLbl
        self._tuImgSize   = tuImgSize
        self._hImgTrns    = hImgTrns
        self._hTgtTrns    = hTgtTrns
        self._hFeatTrns   = hFeatTrns

    def __len__( self: Self ) -> int:

        return self._numSamples

    def __getitem__( self: Self, idx: int ) -> Tuple[Tensor, Tuple[Union[int, Tensor], Tensor]]:

        imgPath = os.path.join(self._datasetPath, self._lFiles[idx])
        tX      = torchvision.io.decode_image(imgPath, 'RGB')
        valY    = self._vLbl[idx]

        if self._hImgTrns:
            tX = self._hImgTrns(tX)

        if isinstance(tX, np.ndarray):
            tY = tX.copy()
        else:
            tY = tX.clone()

        if self._hFeatTrns:
            tX = self._hFeatTrns(tX)

        if self._hTgtTrns:
            tY = self._hTgtTrns(tY)

        return tX, (valY, tY)

    def GetLabels( self: Self ) -> NDArray:

        return self._vLbl

    def GetImageSpatialSize( self: Self ) -> Tuple[int, int]:

        return self._tuImgSize

    def SetTransform( self: Self, trnsType: Literal['Feature', 'Image', 'Target'], hTrns: Optional[Callable] ) -> None:

        match trnsType:
            case 'Feature':
                self._hFeatTrns = hTrns
            case 'Image':
                self._hImgTrns = hTrns
            case 'Target':
                self._hTgtTrns = hTrns
            case _:
                raise ValueError(f'Unsupported transform type: {trnsType}')

# %% Parameters

# Data
folderName = 'CelebAAligned'

# Model
modelName = 'ModelVAECelebA_2026_08_10.pt' #<! OneDrive -> Courses -> Models -> AIProgram
latDim    = 32

# Loss
recLossType = 'MSE'
β           = 1.0

# Training
batchSize  = 128
numWorkers = 4 #<! Number of workers
numEpochs  = 45

# %% Model

class GaussianSamplingLayer(nn.Module):
    def __init__( self ) -> None:
        super().__init__()

    def forward( self, mμ: Tensor, mLogΣ: Tensor ) -> Tensor:

        mσ = torch.exp(0.5 * mLogΣ) #<! The standard deviation from log variance
        mε = torch.randn_like(mσ) #<! Sample random noise from standard normal distribution
        mZ = mμ + mε * mσ #<! Reparameterization trick

        return mZ

class InvertedResidualBlock(nn.Module):
    def __init__( self, numChnlIn: int, numChnlOut: int, expFctr: int = 4, strideSize: int = 1 ) -> None:
        super().__init__()

        self.strideSize = strideSize
        self.enableSkip = (strideSize == 1 and numChnlIn == numChnlOut)
        hiddenDim       = numChnlIn * expFctr

        self.oBlock = nn.Sequential(
            nn.Conv2d(numChnlIn, hiddenDim, 1, bias = False),
            nn.BatchNorm2d(hiddenDim),
            nn.SiLU(),
            nn.Conv2d(hiddenDim, hiddenDim, 3, stride = strideSize, padding = 1, groups = hiddenDim, bias = False),
            nn.BatchNorm2d(hiddenDim),
            nn.SiLU(),
            nn.Conv2d(hiddenDim, numChnlOut, 1, bias = False),
            nn.BatchNorm2d(numChnlOut),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        if self.enableSkip:
            return tX + self.oBlock(tX)
        else:
            return self.oBlock(tX)

class DepthwiseSeparableConv(nn.Module):
    def __init__( self, numChnlIn: int, numChnlOut: int, strideSize: int = 1 ) -> None:
        super().__init__()

        self.strideSize = strideSize

        self.oBlock001 = nn.Sequential(
            nn.Conv2d(numChnlIn, numChnlIn, kernel_size = 3, padding = 1, stride = strideSize, groups = numChnlIn, bias = False),
            nn.BatchNorm2d(numChnlIn),
            nn.SiLU(),
        )
        self.oBlock002 = nn.Sequential(
            nn.Conv2d(numChnlIn, numChnlOut, kernel_size = 1, bias = False),
            nn.BatchNorm2d(numChnlOut),
            nn.SiLU(),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        tX = self.oBlock001(tX)
        tX = self.oBlock002(tX)

        return tX

class DecoderStage(nn.Module):
    def __init__( self, numChnlIn: int, numChnlOut: int, tuOutSize: Tuple[int, int] ) -> None:
        super().__init__()

        self.oStage = nn.Sequential(
            nn.Upsample(size = tuOutSize, mode = 'bilinear', align_corners = False),
            DepthwiseSeparableConv(numChnlIn, numChnlOut),
            InvertedResidualBlock(numChnlOut, numChnlOut, expFctr = 2),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        return self.oStage(tX)

class VariationalAutoEncoder(nn.Module):
    def __init__( self, latDim: int ) -> None:
        super().__init__()

        self.oEncFeatures = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 5, stride = 2, padding = 2, bias = False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            InvertedResidualBlock(32, 48, strideSize = 2),
            InvertedResidualBlock(48, 64, strideSize = 2),
            InvertedResidualBlock(64, 96, strideSize = 2),
            InvertedResidualBlock(96, 128, strideSize = 2),
            InvertedResidualBlock(128, 160),
            InvertedResidualBlock(160, 160),
        )

        self.oEncHead = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(160, 2 * latDim),
        )

        self.oSampler = GaussianSamplingLayer()

        self.oDecInput = nn.Sequential(
            nn.Linear(latDim, 160 * 2 * 2),
            nn.SiLU(),
            nn.Unflatten(dim = 1, unflattened_size = (160, 2, 2)),
        )

        self.oDec = nn.Sequential(
            InvertedResidualBlock(160, 160, expFctr = 2),
            DecoderStage(160, 128, (4, 4)),
            DecoderStage(128, 96, (8, 8)),
            DecoderStage(96, 64, (16, 16)),
            DecoderStage(64, 48, (32, 32)),
            DecoderStage(48, 32, (64, 64)),
            nn.Conv2d(32, 3, kernel_size = 3, padding = 1),
            nn.Sigmoid(),
        )

    def forward( self, tX: Tensor ) -> Tuple[Tensor, Tensor, Tensor]:

        mμ, mLogΣ = self.oEncHead(self.oEncFeatures(tX)).chunk(2, dim = 1)
        if self.training:
            mZ = self.oSampler(mμ, mLogΣ)
        else:
            mZ = mμ
        tXHat = self.oDec(self.oDecInput(mZ))

        return tXHat, mμ, mLogΣ

    def GetEmbedding( self, tX: Tensor ) -> Tensor:

        mμ, _ = self.oEncHead(self.oEncFeatures(tX)).chunk(2, dim = 1)

        return mμ

class VariationalAutoEncoderLoss(nn.Module):
    def __init__( self, recLossType: Literal['MAE', 'MSE'], β: float ) -> None:
        super().__init__()

        match recLossType:
            case 'MAE':
                self.oRecLoss = nn.L1Loss(reduction = 'sum')
            case 'MSE':
                self.oRecLoss = nn.MSELoss(reduction = 'sum')
            case _:
                raise ValueError(f'Unsupported loss type: {recLossType}')

        self.β = β

    def forward( self, tuYHat: Tuple[Tensor, Tensor, Tensor], tuY: Tuple[Tensor, Tensor] ) -> Tensor:

        tXHat, mμ, mLogΣ = tuYHat
        _,      tX        = tuY

        batchSize = tX.size(0)
        recLoss   = self.oRecLoss(tXHat, tX)
        klLoss    = 0.5 * torch.sum(mLogΣ.exp() + mμ.square() - 1.0 - mLogΣ)

        return (recLoss + self.β * klLoss) / batchSize

class AutoEncoderScore(nn.Module):
    def __init__( self ) -> None:
        super().__init__()

    def forward( self, tuYHat: Tuple[Tensor, Tensor, Tensor], tuY: Tuple[Tensor, Tensor] ) -> Tensor:

        tXHat, _, _ = tuYHat
        _, tX       = tuY

        r2Score = r2_score(tXHat.view(-1), tX.view(-1))

        return r2Score

# %% Main Function

def Main( folderName: str, latDim: int, recLossType: Literal['MAE', 'MSE'], β: float, batchSize: int, numWorkers: int, numEpochs: int ) -> None:

    folderPath = os.path.join(DATA_FOLDER_PATH, folderName)
    dsTrain    = CelebADataset(folderPath)

    oTrns = TorchVisionTrns.Compose([
        TorchVisionTrns.RandomHorizontalFlip(p = 0.5),
        TorchVisionTrns.CenterCrop(148),
        TorchVisionTrns.Resize(64),
        TorchVisionTrns.ToDtype(torch.float, scale = True),
    ])
    dsTrain.SetTransform('Image', oTrns)

    persistentWorkers = numWorkers > 0
    dlData = torch.utils.data.DataLoader(dsTrain, batch_size = batchSize, shuffle = True, num_workers = numWorkers, pin_memory = torch.cuda.is_available(), drop_last = True, persistent_workers = persistentWorkers)

    oModel = VariationalAutoEncoder(latDim)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #<! The 1st CUDA device
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Running on device: {runDevice}')
    print(f'Number of samples: {len(dsTrain):,}')
    print(f'Image spatial size: {dsTrain.GetImageSpatialSize()}')

    hL = VariationalAutoEncoderLoss(recLossType, β)
    hS = AutoEncoderScore()
    hL = hL.to(runDevice) #<! Not required!
    hS = hS.to(runDevice)

    oModel = oModel.to(runDevice) #<! Transfer model to device
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = 6e-4, betas = (0.9, 0.99), weight_decay = 1e-3) #<! Define optimizer
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 2e-3, total_steps = numEpochs)

    oModel, lTrainLoss, lTrainScore, lLearnRate = TrainModelDataset(oModel, dlData, oOpt, numEpochs, hL, hS, oSch = oSch)

# %% Main

if __name__ == '__main__':
    Main(folderName, latDim, recLossType, β, batchSize, numWorkers, numEpochs)