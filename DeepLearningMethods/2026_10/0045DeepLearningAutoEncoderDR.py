# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Auto Encoder - Dimensionality Reduction
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.002 | 09/08/2026 | Royi Avital | Added notes on the objective function                              |
# | 1.0.001 | 25/01/2026 | Royi Avital | Simplified the classification head to a linear classifier          |
# | 1.0.000 | 17/01/2026 | Royi Avital | First version                                                      |

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

PROJECT_NAME     = 'FixelCourses'
DATA_FOLDER_NAME = 'DataSets'
BASE_FOLDER_PATH = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)

# %% Course Packages

from DataManipulation import DownloadUrl
from DeepLearningPyTorch import TrainModel

# %% Auxiliary Functions

class MNISTDatasetCSV(torch.utils.data.Dataset):
    """
    MNIST Dataset from CSV File.
    Supports Image Classification and Self Supervised Learning tasks.
    """

    def __init__( self: Self, csvFilePath: str, subSetType: Literal['All', 'Train', 'Val'], *, tuImgSize: Tuple[int, ...] = (28, 28), hImgTrns: Optional[Callable] = None, hFeatTrans: Optional[Callable] = None, hTgtTrns: Optional[Callable] = None ) -> None:
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

        self._dfData    = dfData
        self._dsLbl     = dsLbl
        self._tuImgSize = tuImgSize
        self._hImgTrns  = hImgTrns
        self._hTgtTrns  = hTgtTrns
        self._hFeatTrns = hFeatTrans

    def __len__( self: Self ) -> int:

        return len(self._dfData)

    def __getitem__( self: Self, idx: int ) -> Tuple[Tensor, Tuple[Union[int, Tensor], Tensor]]:

        dsRow = self._dfData.iloc[idx]
        tX    = dsRow.iloc[:-1].to_numpy(np.uint8, copy = True)
        tX    = np.reshape(tX, self._tuImgSize)
        valY  = dsRow.iloc[-1]

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
latDim = 2

# Training
batchSize  = 512
numWorkers = 4 #<! Number of workers
numEpochs  = 45

# %% Model

class AutoEncoder(nn.Module):
    def __init__( self, latDim: int, numCls: int, α: float = 0.1 ):
        super().__init__()

        # Encoder: 1x28x28 -> 64x7x7 -> 2
        self.oEnc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(α, inplace = True),

            nn.Conv2d(16, 32, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(α, inplace = True),

            nn.Conv2d(32, 64, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(α, inplace = True),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.LeakyReLU(α, inplace = True),
            nn.Linear(256, latDim),
        )

        # Decoder: 2 -> 64x7x7 -> 1x28x28
        self.oDec = nn.Sequential(
            nn.Linear(latDim, 256),
            nn.LeakyReLU(α, inplace = True),
            nn.Linear(256, 64 * 7 * 7),
            nn.LeakyReLU(α, inplace = True),
            nn.Unflatten(1, (64, 7, 7)),

            nn.ConvTranspose2d(64, 32, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(α, inplace = True),

            nn.ConvTranspose2d(32, 16, 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(α, inplace = True),

            nn.Conv2d(16, 1, 3, padding = 1),
            nn.Sigmoid(),
        )

        # Classifier
        self.oCls = nn.Sequential(
            nn.Linear(latDim, numCls),
        )

    def forward( self, tX: Tensor ) -> Tuple[Tensor, Tensor]:

        tZ    = self.oEnc(tX) #<! Latent Space (Embedding vector)
        tY    = self.oCls(tZ) #<! Classifier Output (Logits)
        tXHat = self.oDec(tZ) #<! Reconstructed Image

        return tY, tXHat

class AutoEncoderLoss(nn.Module):
    def __init__( self, recLossType: Literal['MAE', 'MSE'], λRec: float, λCls: float ) -> None:
        super().__init__()

        match recLossType:
            case 'MAE':
                self.oRecLoss = nn.L1Loss()
            case 'MSE':
                self.oRecLoss = nn.MSELoss()
            case _:
                raise ValueError(f'Unsupported loss type: {recLossType}')

        self.oClsLoss = nn.CrossEntropyLoss()
        self.λRec     = λRec
        self.λCls     = λCls

    def forward( self, tuYHat: Tuple[Tensor, Tensor], tuY: Tuple[Tensor, Tensor] ) -> Tensor:

        tYHat, tXHat = tuYHat
        tY,    tX    = tuY

        recLoss = self.oRecLoss(tXHat, tX)
        clsLoss = self.oClsLoss(tYHat, tY)

        return self.λRec * recLoss + self.λCls * clsLoss

class AutoEncoderScore(nn.Module):
    def __init__( self ) -> None:
        super().__init__()

    def forward( self, tuYHat: Tuple[Tensor, Tensor], tuY: Tuple[Tensor, Tensor] ) -> Tensor:

        _, tXHat = tuYHat
        _, tX    = tuY

        r2Score = r2_score(tXHat.view(-1), tX.view(-1))

        return r2Score

# %% Main Function

def Main( csvFileName: str, csvFileUrl: str, numCls: int, latDim: int, batchSize: int, numWorkers: int, numEpochs: int ) -> None:

    csvFilePath = os.path.join(DATA_FOLDER_PATH, csvFileName)
    csvFilePath = DownloadUrl(csvFileUrl, csvFilePath)

    dsTrain = MNISTDatasetCSV(csvFilePath, 'Train')
    dsVal   = MNISTDatasetCSV(csvFilePath, 'Val')

    oTrnsTrain = TorchVisionTrns.Compose([
        TorchVisionTrns.ToImage(),
        TorchVisionTrns.ToDtype(torch.float, scale = True),
        TorchVisionTrns.RandomRotation(degrees = 15),
    ])

    oTrnsVal = TorchVisionTrns.Compose([
        TorchVisionTrns.ToImage(),
        TorchVisionTrns.ToDtype(torch.float, scale = True),
    ])

    dsTrain.SetTransform('Image', oTrnsTrain)
    dsVal.SetTransform('Image', oTrnsVal)

    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = batchSize, num_workers = numWorkers)
    dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWorkers)

    oModel = AutoEncoder(latDim, numCls)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #<! The 1st CUDA device

    hL = AutoEncoderLoss('MSE', 0.75, 0.25)
    hS = AutoEncoderScore()
    hL = hL.to(runDevice) #<! Not required!
    hS = hS.to(runDevice)

    oModel = oModel.to(runDevice) #<! Transfer model to device
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = 6e-4, betas = (0.9, 0.99), weight_decay = 1e-3) #<! Define optimizer
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = 2e-2, total_steps = numEpochs)

    oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

# %% Main

if __name__ == '__main__':
    Main(csvFileName, csvFileUrl, numCls, latDim, batchSize, numWorkers, numEpochs)