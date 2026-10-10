# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Generative Models - Variational Auto Encoder
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
import pandas as pd

# Deep Learning
import torch
import torch.nn as nn

from torchvision.transforms import v2 as TorchVisionTrns

from torchmetrics.functional import r2_score

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

L_CLASSES = [str(ii) for ii in range(10)]

PROJECT_NAME       = 'FixelCourses'
DATA_FOLDER_NAME   = 'DataSets'
MODELS_FOLDER_NAME = 'Models'
BASE_FOLDER_PATH   = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH   = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)
MODELS_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, MODELS_FOLDER_NAME)

# %% Course Packages

from DataManipulation import DownloadUrl
from DeepLearningPyTorch import TrainModel

# %% Auxiliary Functions

class MNISTDatasetCSV(torch.utils.data.Dataset):
    """
    MNIST Dataset from CSV File.
    Supports Image Classification and Self Supervised Learning tasks.
    """

    def __init__( self: Self, csvFilePath: str, subSetType: Literal['All', 'Train', 'Val'], *, tuImgSize: Tuple[int, ...] = (28, 28), hImgTrns: Optional[Callable] = None, hFeatTrns: Optional[Callable] = None, hTgtTrns: Optional[Callable] = None ) -> None:
        """
        Constructor Method.

        Parameters
        ----------
        csvFilePath : str
            Path / URL to the CSV file.
        subSetType : Literal['All', 'Train', 'Val']
            Subset type: 'All' for the entire dataset, 'Train' for training set, 'Val' for validation set.
        tuImgSize : Tuple[int, ...], optional
            Image size tuple, by default (28, 28).
        hImgTrns : Optional[Callable], optional
            Transform to be applied to the images, by default None.
        """

        dfData = pd.read_csv(csvFilePath)
        match subSetType:
            case 'All':
                pass
            case 'Train':
                dfData = dfData.iloc[:60000, :].reset_index(drop = True)
            case 'Val':
                dfData = dfData.iloc[60000:, :].reset_index(drop = True)
            case _:
                raise ValueError(f'Unsupported subset type: {subSetType}')

        dsLbl = dfData.iloc[:, -1]
        dsLbl = dsLbl.astype(np.uint8)
        
        # Convert data to NumPy arrays for better performance
        mData = dfData.iloc[:, :-1].to_numpy(np.uint8, copy = True)
        vLbl  = dfData.iloc[:, -1].to_numpy(np.uint8, copy = True)

        self._dfData    = dfData
        self._mData     = mData
        self._dsLbl     = dsLbl
        self._vLbl      = vLbl
        self._tuImgSize = tuImgSize
        self._hImgTrns  = hImgTrns
        self._hTgtTrns  = hTgtTrns
        self._hFeatTrns = hFeatTrns

    def __len__( self: Self ) -> int:

        return len(self._dfData)

    def __getitem__( self: Self, idx: int ) -> Tuple[Tensor, Tuple[Union[int, Tensor], Tensor]]:

        tX    = self._mData[idx]
        tX    = np.reshape(tX, self._tuImgSize)
        valY  = self._vLbl[idx]

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

        return self._dsLbl.to_numpy()

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
csvFileName = 'MNIST.csv'
csvFileUrl  = r'https://huggingface.co/datasets/Royi/MNIST/resolve/main/MNIST.csv'

numCls = len(L_CLASSES) #<! Number of classes

# Model
modelName = 'ModelVariationalAutoEncoder_2026_08_10.pt' #<! OneDrive -> Courses -> Models -> AIProgram
latDim    = 2

# Loss
recLossType = 'MSE'
β           = 1.0

# Training
batchSize  = 512
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

class VariationalAutoEncoder(nn.Module):
    def __init__( self, latDim: int ) -> None:
        super().__init__()

        outDim = 2 * latDim

        # Encoder: 1x28x28 -> `outDim`
        self.oEnc = nn.Sequential(
            nn.Conv2d(1,  8,      kernel_size = 5, bias = False),             nn.BatchNorm2d(8 ), nn.LeakyReLU(),
            nn.Conv2d(8,  16,     kernel_size = 5, bias = False),             nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.Conv2d(16, 32,     kernel_size = 5, bias = False, stride = 2), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 64,     kernel_size = 5, bias = False),             nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, outDim, kernel_size = 4),
            nn.Flatten(),
        )

        self.oSampler = GaussianSamplingLayer()

        # Decoder: `latDim` -> 1x28x28
        self.oDec = nn.Sequential(
            nn.Unflatten(dim = 1, unflattened_size = (latDim, 1, 1)),
            nn.Upsample(scale_factor = 2), nn.Conv2d(latDim, 64, kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Upsample(scale_factor = 2), nn.Conv2d(64,     32, kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Upsample(scale_factor = 2), nn.Conv2d(32,     16, kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.Upsample(scale_factor = 2), nn.Conv2d(16,     8,  kernel_size = 3, padding = 1, bias = False), nn.BatchNorm2d(8 ), nn.LeakyReLU(),
            nn.Upsample(scale_factor = 2), nn.Conv2d(8,      4,  kernel_size = 3, padding = 1, bias = True ),                     nn.LeakyReLU(),
                                           nn.Conv2d(4,      1,  kernel_size = 5, padding = 0, bias = True ),
        )

    def forward( self, tX: Tensor ) -> Tuple[Tensor, Tensor, Tensor]:

        mμ, mLogΣ = self.oEnc(tX).chunk(2, dim = 1) #<! Distribution parameters

        if self.training:
            mZ = self.oSampler(mμ, mLogΣ)
        else:
            mZ = mμ

        tXHat = self.oDec(mZ)

        return tXHat, mμ, mLogΣ

    def GetEmbedding( self, tX: Tensor ) -> Tensor:

        mμ, _ = self.oEnc(tX).chunk(2, dim = 1)

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
        recLoss = self.oRecLoss(tXHat, tX)
        klLoss  = 0.5 * torch.sum(mLogΣ.exp() + mμ.square() - 1.0 - mLogΣ)

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

def Main( csvFileName: str, csvFileUrl: str, latDim: int, recLossType: Literal['MAE', 'MSE'], β: float, batchSize: int, numWorkers: int, numEpochs: int ) -> None:

    csvFilePath = os.path.join(DATA_FOLDER_PATH, csvFileName)
    csvFilePath = DownloadUrl(csvFileUrl, csvFilePath)

    dsTrain = MNISTDatasetCSV(csvFilePath, 'Train')
    dsVal   = MNISTDatasetCSV(csvFilePath, 'Val')

    oTrns = TorchVisionTrns.Compose([
        TorchVisionTrns.ToImage(),
        TorchVisionTrns.ToDtype(torch.float, scale = True),
    ])

    dsTrain.SetTransform('Image', oTrns)
    dsVal.SetTransform('Image', oTrns)

    dlTrain = torch.utils.data.DataLoader(dsTrain, batch_size = batchSize, shuffle = True, num_workers = numWorkers, pin_memory = torch.cuda.is_available(), drop_last = True, persistent_workers = True)
    dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWorkers, pin_memory = torch.cuda.is_available(), drop_last = False, persistent_workers = True)

    oModel = VariationalAutoEncoder(latDim)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #<! The 1st CUDA device
    print(f'CUDA available: {torch.cuda.is_available()}')
    print(f'Running on device: {runDevice}')

    hL = VariationalAutoEncoderLoss(recLossType, β)
    hS = AutoEncoderScore()
    hL = hL.to(runDevice) #<! Not required!
    hS = hS.to(runDevice)

    oModel = oModel.to(runDevice) #<! Transfer model to device
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = 6e-4, betas = (0.9, 0.99), weight_decay = 1e-3) #<! Define optimizer
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 2e-3, total_steps = numEpochs)

    oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

# %% Main

if __name__ == '__main__':
    Main(csvFileName, csvFileUrl, latDim, recLossType, β, batchSize, numWorkers, numEpochs)