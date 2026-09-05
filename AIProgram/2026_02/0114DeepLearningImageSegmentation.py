# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Image to Image Regression - Satellite Aerial to Map
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 05/09/2026 | Royi Avital | First version                                                      |

# %% Packages

# General Tools
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision.io import decode_image
from torchvision.transforms import v2 as TorchVisionTrns

from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.regression import r2_score

# Miscellaneous
import os
import random
import time
from zipfile import ZipFile

# Typing
from typing import Callable, Literal, Optional, Tuple
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

from DataManipulation import DownloadUrl
from DeepLearningPyTorch import GenDataLoaders, TrainModel

# %% Auxiliary Functions

def SSIMScore( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return structural_similarity_index_measure(tYHat, tY, data_range = 1.0)

def ImageR2Score( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return r2_score(tYHat.flatten(), tY.flatten(), multioutput = 'uniform_average')

class SatAerialMapDataset(Dataset):
    """
    Satellite aerial-to-map paired image dataset.
    """

    def __init__( self, rootFolderPath: str, dataSet: Literal['Train', 'Validation', 'All'], /, *, imgSize: Optional[int] = None, hTrns: Optional[Callable] = None ) -> None:
        super().__init__()

        if dataSet not in ('Train', 'Validation', 'All'):
            raise ValueError("dataSet must be 'Train', 'Validation', or 'All'")

        lDataSets = ['Train', 'Validation'] if dataSet == 'All' else [dataSet]
        lImgFiles = []
        for dataSetName in lDataSets:
            dataSetFolderPath = os.path.join(rootFolderPath, dataSetName)
            if not os.path.isdir(dataSetFolderPath):
                raise FileNotFoundError(f'Dataset folder does not exist: {dataSetFolderPath}')

            lDataSetFiles = sorted(
                os.path.join(dataSetFolderPath, fileName)
                for fileName in os.listdir(dataSetFolderPath)
                if os.path.isfile(os.path.join(dataSetFolderPath, fileName)) and fileName.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            lImgFiles.extend(lDataSetFiles)

        self._lImgFiles = lImgFiles
        self._imgSize   = imgSize
        self._hTrns     = hTrns

    def __len__( self ) -> int:

        return len(self._lImgFiles)

    def __getitem__( self, idx: int ) -> Tuple[Tensor, Tensor]:

        tPair = decode_image(self._lImgFiles[idx], mode = 'RGB')
        imgWidthHalf = tPair.shape[2] // 2
        tX = tPair[:, :, :imgWidthHalf]
        tY = tPair[:, :, imgWidthHalf:]

        if self._imgSize is not None:
            tX = TorchVisionTrns.functional.resize(tX, size = (self._imgSize, self._imgSize))
            tY = TorchVisionTrns.functional.resize(tY, size = (self._imgSize, self._imgSize), interpolation = TorchVisionTrns.InterpolationMode.NEAREST)

        if self._hTrns:
            tX = self._hTrns(tX)

        tY = TorchVisionTrns.functional.to_dtype(tY, torch.float, scale = True)

        return tX, tY

    def SetImageSize( self, imgSize: Optional[int] ) -> None:

        self._imgSize = imgSize

    def SetTransforms( self, hTrns: Optional[Callable] ) -> None:

        self._hTrns = hTrns

# %% Parameters

# Data
dataSet    = 'SatAerialToMap'
dataSetUrl = r'https://huggingface.co/datasets/Royi/DataSets/resolve/main/SatAerialToMap.zip'
imgSize    = 256

trainSampleRatio = 0.9
valSampleRatio   = 1 - trainSampleRatio

# Model
modelName      = 'ModelImgSegSatMap_2026_09_05.pt' #<! OneDrive -> Courses -> Models -> AIProgram
numFiltersBase = 16

# Training
lossType   = 'MSE'
scoreType  = 'R2'
batchSize  = 16
numWorkers = 4 #<! Number of workers
numEpochs  = 75

# Optimizer
ηOpt        = 1e-4
tuβ         = (0.9, 0.99)
weightDecay = 5e-5
ηSch        = 7.5e-3

# %% Model

class InvertedResidualBlock(nn.Module):
    def __init__( self, numChnlIn: int, numChnlOut: int, expFctr: int = 4, strideSize: int = 1 ) -> None:
        super().__init__()

        self.enableSkip = (strideSize == 1 and numChnlIn == numChnlOut)
        hiddenDim = numChnlIn * expFctr

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

class µUNet(nn.Module):
    def __init__( self, numChnlIn: int, numChnlOut: int = 3, numFiltersBase: int = 32 ) -> None:
        super().__init__()

        self.oFeatExt = nn.Sequential(
            nn.Conv2d(numChnlIn, numFiltersBase, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase),
            nn.SiLU(),
        )

        self.oEnc001 = InvertedResidualBlock(numFiltersBase, numFiltersBase * 2, strideSize = 2)
        self.oEnc002 = InvertedResidualBlock(numFiltersBase * 2, numFiltersBase * 4, strideSize = 2)
        self.oEnc003 = InvertedResidualBlock(numFiltersBase * 4, numFiltersBase * 8, strideSize = 2)
        self.oEnc004 = InvertedResidualBlock(numFiltersBase * 8, numFiltersBase * 16, strideSize = 2)

        self.oEmbed = InvertedResidualBlock(numFiltersBase * 16, numFiltersBase * 16, strideSize = 1)

        self.oUp004 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(numFiltersBase * 16, numFiltersBase * 16, 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 16),
            nn.SiLU(),
        )
        self.oDec004 = nn.Sequential(
            nn.Conv2d(numFiltersBase * 16 + numFiltersBase * 8, numFiltersBase * 8, 1),
            InvertedResidualBlock(numFiltersBase * 8, numFiltersBase * 8),
        )

        self.oUp003 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(numFiltersBase * 8, numFiltersBase * 8, 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 8),
            nn.SiLU(),
        )
        self.oDec003 = nn.Sequential(
            nn.Conv2d(numFiltersBase * 8 + numFiltersBase * 4, numFiltersBase * 4, 1),
            InvertedResidualBlock(numFiltersBase * 4, numFiltersBase * 4),
        )

        self.oUp002 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(numFiltersBase * 4, numFiltersBase * 4, 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 4),
            nn.SiLU(),
        )
        self.oDec002 = nn.Sequential(
            nn.Conv2d(numFiltersBase * 4 + numFiltersBase * 2, numFiltersBase * 2, 1),
            InvertedResidualBlock(numFiltersBase * 2, numFiltersBase * 2),
        )

        self.oUp001 = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(numFiltersBase * 2, numFiltersBase * 2, 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 2),
            nn.SiLU(),
        )

        self.oHeadImg = nn.Sequential(
            nn.Conv2d(numFiltersBase * 2 + numFiltersBase, numFiltersBase * 2, 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 2),
            nn.SiLU(),
            nn.Conv2d(numFiltersBase * 2, numChnlOut, 1),
            nn.Sigmoid(),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        tX0 = self.oFeatExt(tX)
        tX1 = self.oEnc001(tX0)
        tX2 = self.oEnc002(tX1)
        tX3 = self.oEnc003(tX2)
        tX4 = self.oEnc004(tX3)

        tEm = self.oEmbed(tX4)

        tD4 = self.oUp004(tEm)
        tD4 = self.oDec004(torch.cat([tD4, tX3], dim = 1))
        tD3 = self.oUp003(tD4)
        tD3 = self.oDec003(torch.cat([tD3, tX2], dim = 1))
        tD2 = self.oUp002(tD3)
        tD2 = self.oDec002(torch.cat([tD2, tX1], dim = 1))
        tD1 = self.oUp001(tD2)
        tY = self.oHeadImg(torch.cat([tD1, tX0], dim = 1))

        return tY

class Pix2PixLoss(nn.Module):
    def __init__( self, lossType: Literal['L1', 'L2', 'MSE'] = 'MSE' ) -> None:
        super().__init__()

        match lossType:
            case 'L1':
                self.oLoss = nn.L1Loss()
            case 'L2' | 'MSE':
                self.oLoss = nn.MSELoss()
            case _:
                raise ValueError('The parameter `lossType` must be either `L1`, `L2` or `MSE`')

    def forward( self, tYHat: Tensor, tY: Tensor ) -> Tensor:

        return self.oLoss(tYHat, tY)

class Pix2PixScore(nn.Module):
    def __init__( self, scoreType: Literal['SSIM', 'R2'] = 'SSIM' ) -> None:
        super().__init__()

        match scoreType:
            case 'SSIM':
                self.hScore = SSIMScore
            case 'R2':
                self.hScore = ImageR2Score
            case _:
                raise ValueError('The parameter `scoreType` must be either `SSIM` or `R2`')

    def forward( self, tYHat: Tensor, tY: Tensor ) -> Tensor:

        return self.hScore(tYHat, tY)

# %% Main Function

def Main( dataSet: str, dataSetUrl: str, imgSize: int, trainSampleRatio: float, valSampleRatio: float, numFiltersBase: int, lossType: Literal['L1', 'L2', 'MSE'], scoreType: Literal['SSIM', 'R2'], batchSize: int, numWorkers: int, numEpochs: int, ηOpt: float, tuβ: Tuple[float, float], weightDecay: float, ηSch: float ) -> None:

    datasetFolderPath = os.path.join(DATA_FOLDER_PATH, dataSet)
    if not os.path.isdir(datasetFolderPath):
        fileName = os.path.join(DATA_FOLDER_PATH, f'{dataSet}.zip')
        DownloadUrl(dataSetUrl, fileName)
        with ZipFile(fileName, 'r') as zipFile:
            zipFile.extractall(DATA_FOLDER_PATH) #<! The Zip file contains a folder
        time.sleep(1.0) #<! Wait for the file system to update
        os.remove(fileName)

    oTrnsTrain = TorchVisionTrns.Compose([
        TorchVisionTrns.ToDtype(torch.float, scale = True),
        TorchVisionTrns.RandomChoice([
            TorchVisionTrns.RandomGrayscale(p = 1.0),
            TorchVisionTrns.GaussianBlur(7, sigma = (0.1, 1.0)),
            TorchVisionTrns.RandomEqualize(p = 1.0),
            TorchVisionTrns.RandomAutocontrast(p = 1.0),
            TorchVisionTrns.GaussianNoise(sigma = 0.05),
            TorchVisionTrns.RandomErasing(p = 1.0, scale = (0.05, 0.15), ratio = (0.5, 2.0), value = 0, inplace = True),
            TorchVisionTrns.RGB(),
        ], p = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.58]),
    ])

    oTrnsVal = TorchVisionTrns.Compose([
        TorchVisionTrns.ToDtype(torch.float, scale = True),
    ])
    dsTrain = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize, hTrns = oTrnsTrain)
    dsVal   = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize, hTrns = oTrnsVal)

    numSamples = len(dsTrain)
    vIdxTrain, vIdxVal = train_test_split(np.arange(numSamples), test_size = valSampleRatio, train_size = trainSampleRatio, random_state = seedNum, shuffle = True)
    dsTrain = torch.utils.data.Subset(dsTrain, vIdxTrain)
    dsVal   = torch.utils.data.Subset(dsVal, vIdxVal)

    dlTrain, dlVal = GenDataLoaders(dsTrain, dsVal, batchSize, numWorkers = numWorkers, pinMemory = torch.cuda.is_available(), persWork = numWorkers > 0)

    oModel = µUNet(numChnlIn = 3, numChnlOut = 3, numFiltersBase = numFiltersBase)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f'Running on device: {runDevice}')
    print(f'The training data set contains  : {len(dsTrain):4d} samples.')
    print(f'The validation data set contains: {len(dsVal):4d} samples.')

    hL = Pix2PixLoss(lossType = lossType)
    hS = Pix2PixScore(scoreType = scoreType)
    hL = hL.to(runDevice)
    hS = hS.to(runDevice)

    oModel = oModel.to(runDevice)
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = ηOpt, betas = tuβ, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = ηSch, total_steps = numEpochs)

    oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

# %% Main

if __name__ == '__main__':
    Main(dataSet, dataSetUrl, imgSize, trainSampleRatio, valSampleRatio, numFiltersBase, lossType, scoreType, batchSize, numWorkers, numEpochs, ηOpt, tuβ, weightDecay, ηSch)