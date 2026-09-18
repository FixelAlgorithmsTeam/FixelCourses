# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Image to Image - Image Segmentation with U-Net
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.1.000 | 19/09/2026 | Royi Avital | Updated the model structure for a smaller model                    |
# | 1.0.003 | 18/09/2026 | Royi Avital | Fixed dimension mismatch in the model                              |
# | 1.0.002 | 02/02/2026 | Royi Avital | Expanded the information on the Inverted Residual Block            |
# | 1.0.001 | 01/02/2026 | Royi Avital | Simplified the classification head                                 |
# | 1.0.000 | 21/01/2026 | Royi Avital | First version                                                      |

# %% Packages

# General Tools
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.classification import multiclass_f1_score
import torchvision
from torchvision.io import decode_image
from torchvision.transforms import v2 as TorchVisionTrns

# Miscellaneous
import os
import random

# Typing
from typing import Callable, Dict, List, Optional, Tuple
from torch import Tensor

# %% Constants

DATA_FOLDER_NAME = 'DataSets'
BASE_FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)

# %% Course Packages

from DeepLearningPyTorch import GenDataLoaders, TrainModel

# %% Auxiliary Functions

class MSDDataset(Dataset):
    """Mobile Phone Defect Segmentation Dataset with paired image/mask transforms."""

    def __init__( self, imgFolderPath: str, maskFolderPath: str, dCls: Dict[int, str], /, *, hTrns: Optional[Callable] = None, lImgFormats: Tuple[str, ...] = ('jpg', 'jpeg', 'png') ) -> None:
        super().__init__()

        lImgFiles = os.listdir(imgFolderPath)
        lImgFiles = [fileName for fileName in lImgFiles if (fileName.split('.')[-1].lower() in lImgFormats) and (os.path.isfile(os.path.join(maskFolderPath, fileName.split('.')[0] + '.png')))]
        lMskFiles = [fileName.split('.')[0] + '.png' for fileName in lImgFiles]

        self._imgFolderPath  = imgFolderPath
        self._maskFolderPath = maskFolderPath
        self._dCls           = dCls
        self._dClsIdx        = {clsName: clsIdx for clsIdx, clsName in dCls.items()}
        self._hTrns          = hTrns
        self._lImgFiles      = lImgFiles
        self._lMskFiles      = lMskFiles
        self._lImgCls        = [self._ParseCls(fileName) for fileName in lImgFiles]
        self._numFiles       = len(lImgFiles)

    def __len__( self ) -> int:

        return self._numFiles

    def __getitem__( self, idx: int ) -> Tuple[Tensor, Tensor]:

        imgPath = os.path.join(self._imgFolderPath, self._lImgFiles[idx])
        mskPath = os.path.join(self._maskFolderPath, self._lMskFiles[idx])

        tI = decode_image(imgPath, mode = 'RGB')
        tM = decode_image(mskPath, mode = 'RGB')
        clsLbl = self._lImgCls[idx]

        tB = (tM.sum(dim = 0) > 0).to(torch.long)
        tM = clsLbl * tB

        tI = torchvision.tv_tensors.Image(tI, dtype = torch.uint8)
        tM = torchvision.tv_tensors.Mask(tM, dtype = torch.long)

        if self._hTrns:
            tI, tM = self._hTrns(tI, tM)

        return tI, tM

    def _ParseCls( self, fileName: str ) -> int:

        clsStr = fileName.split('_')[0]
        clsLbl = self._dClsIdx[clsStr]

        return clsLbl

    def SetTransforms( self, hTrns: Callable ) -> None:

        self._hTrns = hTrns

    def GetClasses( self ) -> Dict[int, str]:

        return self._dCls.copy()

    def GetLabels( self ) -> List[int]:

        return self._lImgCls.copy()

# %% Parameters

seedNum = 512

# Data
folderName = 'MSD'
dCls       = {0: 'None', 1: 'Scratch', 2: 'Stain', 3: 'Oil'}
tuImgSize  = (320, 576)

# Model
numFiltersBase = 4
weightSeg      = 3.0
weightCls      = 1.0
segThr         = 0.5

# Training
trainSampleRatio = 0.9
batchSize        = 8
numWorkers       = 4
numEpochs        = 35

# Optimizer
ηOpt        = 1e-4
tuβ         = (0.9, 0.99)
weightDecay = 5e-5
ηSch        = 7.5e-5

# %% Model

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

class µSegmentor(nn.Module):
    def __init__( self, numChnlIn: int, numCls: int, numFiltersBase: int = 8 ) -> None:
        super().__init__()

        self.oFeatExt = nn.Sequential(
            nn.Conv2d(numChnlIn, numFiltersBase, 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase),
            nn.SiLU(),
        )

        self.oEnc001 = InvertedResidualBlock(numFiltersBase    , numFiltersBase * 2, strideSize = 2)
        self.oEnc002 = InvertedResidualBlock(numFiltersBase * 2, numFiltersBase * 4, strideSize = 2)
        self.oEnc003 = InvertedResidualBlock(numFiltersBase * 4, numFiltersBase * 8, strideSize = 2)
        self.oEnc004 = InvertedResidualBlock(numFiltersBase * 8, numFiltersBase * 8, strideSize = 2)

        self.oEmbed = InvertedResidualBlock(numFiltersBase * 8, numFiltersBase * 16, strideSize = 1)

        self.oDec004 = nn.Sequential(
            nn.ConvTranspose2d(numFiltersBase * 16, numFiltersBase * 16, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 16),
            nn.SiLU(),
            InvertedResidualBlock(numFiltersBase * 16, numFiltersBase * 8),
        )

        self.oDec003 = nn.Sequential(
            nn.ConvTranspose2d(numFiltersBase * 16, numFiltersBase * 8, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 8),
            nn.SiLU(),
            InvertedResidualBlock(numFiltersBase * 8, numFiltersBase * 4),
        )

        self.oDec002 = nn.Sequential(
            nn.ConvTranspose2d(numFiltersBase * 8, numFiltersBase * 2, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False), #<! Optimize memory at high resolution by low number of channels
            nn.BatchNorm2d(numFiltersBase * 2),
            nn.SiLU(),
            InvertedResidualBlock(numFiltersBase * 2, numFiltersBase * 2, expFctr = 2),
        )

        self.oDec001 = nn.Sequential(
            nn.ConvTranspose2d(numFiltersBase * 4, numFiltersBase, kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = False), #<! Optimize memory at high resolution by low number of channels
            nn.BatchNorm2d(numFiltersBase),
            nn.SiLU(),
            InvertedResidualBlock(numFiltersBase, numFiltersBase, expFctr = 2),
        )

        self.oHeadMask = nn.Sequential(
            nn.Conv2d(numFiltersBase * 2, numFiltersBase * 2, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(numFiltersBase * 2),
            nn.SiLU(),
            nn.Conv2d(numFiltersBase * 2, 1, 1),
        )

        self.oHeadCls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(numFiltersBase * 16, numCls),
        )

    def forward( self, tX: Tensor ) -> Tuple[Tensor, Tensor]:

        tX0 = self.oFeatExt(tX)
        tX1 = self.oEnc001(tX0)
        tX2 = self.oEnc002(tX1)
        tX3 = self.oEnc003(tX2)
        tX4 = self.oEnc004(tX3)

        tEm = self.oEmbed(tX4)

        tD4 = self.oDec004(tEm)
        if tD4.shape[2:] != tX3.shape[2:]:
            tD4 = F.interpolate(tD4, size = tX3.shape[2:], mode = 'bilinear', align_corners = False)
        tD4 = torch.cat([tD4, tX3], dim = 1)

        tD3 = self.oDec003(tD4)
        if tD3.shape[2:] != tX2.shape[2:]:
            tD3 = F.interpolate(tD3, size = tX2.shape[2:], mode = 'bilinear', align_corners = False)
        tD3 = torch.cat([tD3, tX2], dim = 1)

        tD2 = self.oDec002(tD3)
        if tD2.shape[2:] != tX1.shape[2:]:
            tD2 = F.interpolate(tD2, size = tX1.shape[2:], mode = 'bilinear', align_corners = False)
        tD2 = torch.cat([tD2, tX1], dim = 1)

        tD1 = self.oDec001(tD2)
        if tD1.shape[2:] != tX0.shape[2:]:
            tD1 = F.interpolate(tD1, size = tX0.shape[2:], mode = 'bilinear', align_corners = False)
        tD1 = torch.cat([tD1, tX0], dim = 1)

        tO = self.oHeadMask(tD1)
        tY = self.oHeadCls(tEm)

        return tO, tY

class SegmentationClsLoss(nn.Module):
    def __init__( self, weightSeg: float = 1.0, weightCls: float = 1.0 ) -> None:
        super().__init__()

        self.weightSeg = weightSeg
        self.weightCls = weightCls
        self.oSegLoss = nn.BCEWithLogitsLoss()
        self.oClsLoss = nn.CrossEntropyLoss()

    def forward( self, tuZ: Tuple[Tensor, Tensor], tMTgt: Tensor ) -> Tensor:

        tM   = tuZ[0].squeeze(1)
        mCls = tuZ[1]

        vClsTgt = tMTgt.view(tMTgt.shape[0], -1).amax(dim = 1)
        tMTgt   = tMTgt.not_equal(0).to(torch.float32)

        segLoss = self.oSegLoss(tM, tMTgt)
        clsLoss = self.oClsLoss(mCls, vClsTgt)
        valLoss = self.weightSeg * segLoss + self.weightCls * clsLoss

        return valLoss

class SegmentationClsScore(nn.Module):
    def __init__( self, numCls: int, segThr: float, weightSeg: float = 1.0, weightCls: float = 1.0 ) -> None:
        super().__init__()

        self.numCls    = numCls
        self.segThr    = segThr
        self.weightSeg = weightSeg / (weightSeg + weightCls)
        self.weightCls = weightCls / (weightSeg + weightCls)

    def forward( self, tuZ: Tuple[Tensor, Tensor], tMTgt: Tensor ) -> Tensor:

        tM   = tuZ[0]
        mCls = tuZ[1]
        tB   = (torch.sigmoid(tM.squeeze(1)) > self.segThr).to(torch.long)
        vCls = mCls.argmax(dim = 1)

        vClsTgt = tMTgt.view(tMTgt.shape[0], -1).amax(dim = 1)
        tBTgt   = tMTgt.not_equal(0).to(torch.long)

        segScore = mean_iou(tB, tBTgt, num_classes = 2, include_background = True, per_class = False, input_format = 'index').mean()
        clsScore = multiclass_f1_score(vCls, vClsTgt, num_classes = self.numCls, average = 'macro', top_k = 1, multidim_average = 'global', ignore_index = None, validate_args = False, zero_division = 0)
        valScore = self.weightSeg * segScore + self.weightCls * clsScore

        return valScore

# %% Main Function

def Main( folderPath: str, dCls: Dict[int, str], tuImgSize: Tuple[int, int], numFiltersBase: int,
          weightSeg: float, weightCls: float, segThr: float, trainSampleRatio: float,
          batchSize: int, numWorkers: int, numEpochs: int, ηOpt: float,
          tuβ: Tuple[float, float], weightDecay: float, ηSch: float, seedNum: int ) -> None:

    np.random.seed(seedNum)
    random.seed(seedNum)
    torch.backends.cudnn.benchmark = True

    imgFolderPath  = os.path.join(folderPath, 'Images')
    maskFolderPath = os.path.join(folderPath, 'Masks')
    if not os.path.isdir(imgFolderPath) or not os.path.isdir(maskFolderPath):
        raise FileNotFoundError(f'Prepare the MSD Images and Masks folders using 0113DeepLearningImageSegmentation.ipynb first: {folderPath}')

    oTrns = TorchVisionTrns.Compose([
        TorchVisionTrns.Resize(tuImgSize),
        TorchVisionTrns.ToDtype(torch.float, scale = True),
        TorchVisionTrns.RandomChoice([
            TorchVisionTrns.RandomGrayscale(p = 1.0),
            TorchVisionTrns.RandomHorizontalFlip(p = 1.0),
            TorchVisionTrns.RandomVerticalFlip(p = 1.0),
            TorchVisionTrns.RandomRotation(degrees = 10),
            TorchVisionTrns.RGB(),
        ], p = [0.15, 0.15, 0.15, 0.15, 0.40]),
    ])

    dsData     = MSDDataset(imgFolderPath, maskFolderPath, dCls, hTrns = oTrns)
    numSamples = len(dsData)
    lLabels    = dsData.GetLabels()
    vIdxTrain, vIdxVal = train_test_split(np.arange(numSamples), test_size = 1 - trainSampleRatio, train_size = trainSampleRatio, random_state = seedNum, shuffle = True, stratify = lLabels)
    dsTrain = torch.utils.data.Subset(dsData, vIdxTrain)
    dsVal   = torch.utils.data.Subset(dsData, vIdxVal)

    persistentWorkers = numWorkers > 0
    pinMemory = torch.cuda.is_available()
    dlTrain, dlVal = GenDataLoaders(dsTrain, dsVal, batchSize, numWorkers = numWorkers, pinMemory = pinMemory, persWork = persistentWorkers)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    numCls = len(dCls)
    oModel = µSegmentor(numChnlIn = 3, numCls = numCls, numFiltersBase = numFiltersBase).to(runDevice)
    hL = SegmentationClsLoss(weightSeg, weightCls).to(runDevice)
    hS = SegmentationClsScore(numCls, segThr, weightSeg, weightCls).to(runDevice)

    print(f'Running on device: {runDevice}')
    print(f'The training data set contains  : {len(dsTrain):4d} samples.')
    print(f'The validation data set contains: {len(dsVal):4d} samples.')
    print(f'Image spatial size: {tuImgSize}')
    print(f'Workers per loader: {numWorkers}, Persistent workers: {persistentWorkers}, Pinned memory: {pinMemory}')

    oOpt = torch.optim.AdamW(oModel.parameters(), lr = ηOpt, betas = tuβ, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = ηSch, total_steps = numEpochs)

    oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

# %% Main

if __name__ == '__main__':
    Main(os.path.join(DATA_FOLDER_PATH, folderName), dCls, tuImgSize, numFiltersBase,
         weightSeg, weightCls, segThr, trainSampleRatio, batchSize, numWorkers,
         numEpochs, ηOpt, tuβ, weightDecay, ηSch, seedNum)