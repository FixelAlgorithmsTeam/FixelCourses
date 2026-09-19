# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Deep Learning - Computer Vision - Object Detection
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.1.000 | 19/09/2026 | Royi Avital | Changed the model to a smaller model                               |
# | 1.0.001 | 15/09/2026 | Royi Avital | Added visualization of the model                                   |
# | 1.0.000 | 16/06/2024 | Royi Avital | First version                                                      |

# %% Packages

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, default_collate
import torchvision

import random

from typing import Dict, List, Tuple, Union
from torch import Tensor

# %% Course Packages

from DeepLearningObjectDetection import BBoxFormat, GenLabeldEllipseImg
from DeepLearningPyTorch import GenDataLoaders, TrainModel

# %% Constants

L_CLASSES  = ['R', 'G', 'B']
T_IMG_SIZE = (100, 100, 3)

# %% Auxiliary Functions

class ObjectDetectionDataset(Dataset):
    """Fixed synthetic samples in shared CPU storage for Windows workers.

    The generator produces binary RGB images, so uint8 storage is lossless.
    Images are converted to float32 during collation without rescaling.
    """

    def __init__( self, numSamples: int, tuImgSize: Tuple[int, int, int], maxObj: int ) -> None:
        super().__init__()

        if numSamples <= 0 or maxObj <= 0:
            raise ValueError('numSamples and maxObj must be positive.')
        if len(tuImgSize) != 3 or tuImgSize[2] != 3 or min(tuImgSize[:2]) < 10:
            raise ValueError('tuImgSize must be (height, width, 3), with height and width at least 10.')

        # Share memory to prevent copies in multi process data loading
        self._tX = torch.empty((numSamples, 3, tuImgSize[0], tuImgSize[1]), dtype = torch.uint8).share_memory_()
        self._mY = torch.zeros((numSamples, maxObj), dtype = torch.long).share_memory_()
        self._tB = torch.zeros((numSamples, maxObj, 4), dtype = torch.float32).share_memory_()
        self._vNumObj = torch.empty(numSamples, dtype = torch.long).share_memory_()

        for sampleIdx in range(numSamples):
            numObj = np.random.randint(maxObj) + 1
            mI, vLbl, mBB = GenLabeldEllipseImg(tuImgSize[:2], numObj, boxFormat = BBoxFormat.YOLO)
            self._tX[sampleIdx].copy_(torch.from_numpy(np.transpose(mI, (2, 0, 1))))
            self._mY[sampleIdx, :numObj].copy_(torch.from_numpy(vLbl))
            self._tB[sampleIdx, :numObj].copy_(torch.from_numpy(mBB))
            self._vNumObj[sampleIdx] = numObj

    def __len__( self ) -> int:

        return self._tX.shape[0]

    def __getitem__( self, idx: int ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        numObj = int(self._vNumObj[idx])
        return self._tX[idx], (self._mY[idx, :numObj], self._tB[idx, :numObj])


def CollateSyntheticDetection( lBatch: List[Tuple[Tensor, Tuple[Tensor, Tensor]]] ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:

    tX = default_collate([sample[0] for sample in lBatch]).to(torch.float32)
    lY = [{'Labels': sample[1][0], 'Boxes': sample[1][1]} for sample in lBatch]

    return tX, lY

# %% Parameters

seedNum = 512

numSamplesTrain = 30_000
numSamplesVal   = 10_000
maxObj          = 3
gridSize        = 5

weightObj      = 1.0
weightReg      = 1.0
weightCls      = 1.0
probThr        = 0.5
iouThr         = 0.5
labelSmoothing = 0.1

batchSize  = 128
numWorkers = 6
numEpochs  = 35

learnRate    = 1e-5
tuBetas      = (0.9, 0.99)
weightDecay  = 1e-5
maxLearnRate = 2.5e-3

# %% Model

class DepthWiseSeparableConv2D(nn.Module):
    def __init__( self, inChannels: int, outChannels: int, kernelSize: int, stride: int = 1, padding: Union[int, str] = 0, bias: bool = True, *, dilation: int = 1 ) -> None:
        super().__init__()

        self.oDepthWiseSeparableConv2D = nn.Sequential(
            nn.Conv2d(inChannels, inChannels, kernel_size = kernelSize, stride = stride, padding = padding, dilation = dilation, groups = inChannels, bias = bias),
            nn.Conv2d(inChannels, outChannels, kernel_size = 1, stride = 1, padding = 0, bias = bias),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        return self.oDepthWiseSeparableConv2D(tX)


class µDetector(nn.Module):
    def __init__( self, numClasses: int, gridSize: int ) -> None:
        super().__init__()

        self.numClasses = numClasses
        self.gridSize   = gridSize

        self.oFeatureExtractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(16), nn.ReLU(),
            DepthWiseSeparableConv2D(16, 32, 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(),
            DepthWiseSeparableConv2D(32, 64, 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(64), nn.ReLU(),
            DepthWiseSeparableConv2D(64, 64, 3, stride = 1, padding = 2, bias = False, dilation = 2), nn.BatchNorm2d(64), nn.ReLU(),
            DepthWiseSeparableConv2D(64, 64, 3, stride = 1, padding = 4, bias = False, dilation = 4), nn.BatchNorm2d(64), nn.ReLU(),
        )

        self.oClassifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((gridSize, gridSize)),
            nn.Conv2d(64, 5 + numClasses, 1),
        )

    def forward( self, tX: Tensor ) -> Tensor:

        tX = self.oFeatureExtractor(tX)
        tX = self.oClassifier(tX)
        tX = torch.cat((tX[:, :1], torch.sigmoid(tX[:, 1:5]), tX[:, 5:]), dim = 1)

        return tX


class ObjDetLoss(nn.Module):
    def __init__( self, numCls: int, weightObj: float, weightReg: float, weightCls: float, labelSmoothing: float = 0.0 ) -> None:
        super().__init__()

        self.numCls    = numCls
        self.weightObj = weightObj
        self.weightReg = weightReg
        self.weightCls = weightCls
        self.oObjLoss  = nn.BCEWithLogitsLoss()
        self.oRegLoss  = nn.MSELoss()
        self.oClsLoss  = nn.CrossEntropyLoss(label_smoothing = labelSmoothing)

    def forward( self, tYHat: Tensor, lY: List[Dict[str, Tensor]] ) -> Tensor:

        batchSize = tYHat.shape[0]
        gridSize  = tYHat.shape[2]
        runDevice = tYHat.device

        tObjTgt = torch.zeros((batchSize, gridSize, gridSize), device = runDevice, dtype = tYHat.dtype)
        tRegTgt = torch.zeros((batchSize, gridSize, gridSize, 4), device = runDevice, dtype = tYHat.dtype)
        tClsTgt = torch.zeros((batchSize, gridSize, gridSize), device = runDevice, dtype = torch.long)

        for sampleIdx in range(batchSize):
            tB = lY[sampleIdx]['Boxes']
            tC = lY[sampleIdx]['Labels']

            if tB.numel() == 0:
                continue

            vCxNorm, vCyNorm, vW, vH = tB.T
            vCellX = (vCxNorm * gridSize).floor().to(torch.long).clamp_(0, gridSize - 1)
            vCellY = (vCyNorm * gridSize).floor().to(torch.long).clamp_(0, gridSize - 1)
            vCxRel = vCxNorm * gridSize - vCellX.to(vCxNorm.dtype)
            vCyRel = vCyNorm * gridSize - vCellY.to(vCyNorm.dtype)

            tObjTgt[sampleIdx, vCellY, vCellX]    = 1.0
            tRegTgt[sampleIdx, vCellY, vCellX, :] = torch.stack((vCxRel, vCyRel, vW, vH), dim = 1)
            tClsTgt[sampleIdx, vCellY, vCellX]    = tC.to(torch.long)

        objLoss = self.oObjLoss(tYHat[:, 0, :, :], tObjTgt)

        mObjMask = tObjTgt.to(torch.bool)
        if mObjMask.any():
            tYHatReg = tYHat[:, 1:5, :, :].permute(0, 2, 3, 1)
            regLoss  = self.oRegLoss(tYHatReg[mObjMask], tRegTgt[mObjMask])
            tYHatCls = tYHat[:, 5:, :, :].permute(0, 2, 3, 1)
            clsLoss  = self.oClsLoss(tYHatCls[mObjMask], tClsTgt[mObjMask])
        else:
            regLoss = 0.0
            clsLoss = 0.0

        return (self.weightObj * objLoss) + (self.weightReg * regLoss) + (self.weightCls * clsLoss)


class ObjDetScore(nn.Module):
    def __init__( self, numCls: int, gridSize: int, probThr: float, iouThr: float, runDevice: torch.device ) -> None:
        super().__init__()

        self.numCls    = numCls
        self.gridSize  = gridSize
        self.probThr   = probThr
        self.iouThr    = iouThr
        self.runDevice = runDevice

    def forward( self, tYHat: Tensor, lY: List[Dict[str, Tensor]] ) -> float:

        batchSize = tYHat.shape[0]
        gridSize  = tYHat.shape[2]
        numObjDet = 0
        numMatch  = 0

        for sampleIdx in range(batchSize):
            mB = lY[sampleIdx]['Boxes']
            vC = lY[sampleIdx]['Labels']
            mPredMask = torch.sigmoid(tYHat[sampleIdx, 0, :, :]).gt(self.probThr)

            numObj  = int(mB.shape[0])
            numPred = int(mPredMask.sum().item())
            numObjDet += numObj + numPred

            if (numObj == 0) or (numPred == 0):
                continue

            mIdx = torch.nonzero(mPredMask, as_tuple = False)
            vPy, vPx = mIdx[:, 0], mIdx[:, 1]
            tClsP = tYHat[sampleIdx, 5:, vPy, vPx]
            vClsP = torch.argmax(tClsP, dim = 0).to(torch.long)

            mBP = tYHat[sampleIdx, 1:5, vPy, vPx].T.clone()
            mBP[:, 0] = (vPx.to(mBP.dtype) + mBP[:, 0]) / gridSize
            mBP[:, 1] = (vPy.to(mBP.dtype) + mBP[:, 1]) / gridSize

            mIoU = torchvision.ops.box_iou(mBP, mB, 'cxcywh')
            vC = vC.to(torch.long)
            mMatchMat = (mIoU > self.iouThr) & (vClsP[:, None] == vC[None, :])

            numMatchBatchPredGt = int(mMatchMat.any(dim = 1).sum().item())
            numMatchBatchGtPred = int(mMatchMat.any(dim = 0).sum().item())
            numMatch += numMatchBatchPredGt + numMatchBatchGtPred

        return float(numMatch / numObjDet) if numObjDet > 0 else 0.0

# %% Main Function

def Main( numSamplesTrain: int, numSamplesVal: int, tuImgSize: Tuple[int, int, int], maxObj: int,
          numCls: int, gridSize: int, weightObj: float, weightReg: float, weightCls: float,
          probThr: float, iouThr: float, labelSmoothing: float, batchSize: int, numWorkers: int,
          numEpochs: int, learnRate: float, tuBetas: Tuple[float, float], weightDecay: float,
          maxLearnRate: float, seedNum: int ) -> None:

    if batchSize <= 0 or numSamplesTrain < batchSize:
        raise ValueError('numSamplesTrain must be at least one positive training batch.')

    np.random.seed(seedNum)
    random.seed(seedNum)
    torch.manual_seed(seedNum)
    torch.backends.cudnn.benchmark = True

    print(f'Generating {numSamplesTrain} training and {numSamplesVal} validation samples...', flush = True)
    dsTrain = ObjectDetectionDataset(numSamplesTrain, tuImgSize, maxObj)
    dsVal   = ObjectDetectionDataset(numSamplesVal, tuImgSize, maxObj)

    persistentWorkers = numWorkers > 0
    pinMemory = torch.cuda.is_available()
    dlTrain, dlVal = GenDataLoaders(dsTrain, dsVal, batchSize, numWorkers = numWorkers,
                                  CollateFn = CollateSyntheticDetection, pinMemory = pinMemory,
                                  persWork = persistentWorkers)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    oModel = µDetector(numClasses = numCls, gridSize = gridSize).to(runDevice)
    hL = ObjDetLoss(numCls, weightObj, weightReg, weightCls, labelSmoothing).to(runDevice)
    hS = ObjDetScore(numCls, gridSize, probThr, iouThr, runDevice).to(runDevice)

    print(f'Running on device: {runDevice}')
    print(f'The training data set contains  : {len(dsTrain):5d} samples.')
    print(f'The validation data set contains: {len(dsVal):5d} samples.')
    print(f'Image spatial size: {tuImgSize[:2]}, Grid size: {gridSize}')
    print(f'Model parameters: {sum(parameter.numel() for parameter in oModel.parameters()):,}')
    print(f'Workers per loader: {numWorkers}, Persistent workers: {persistentWorkers}, Pinned memory: {pinMemory}')
    print(f'Prefetch factor: {dlTrain.prefetch_factor}, Shared image storage: uint8')

    oOpt = torch.optim.AdamW(oModel.parameters(), lr = learnRate, betas = tuBetas, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = maxLearnRate, total_steps = numEpochs)

    _, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

    # Update the saved data with hyper parameters and training history
    dCheckPoint = torch.load('BestModel.pt', map_location = 'cpu', weights_only = True)
    dCheckPoint.update({
        'ModelParams': {'numClasses': numCls, 'gridSize': gridSize},
        'TrainParams': {
            'numSamplesTrain': numSamplesTrain,
            'numSamplesVal': numSamplesVal,
            'tuImgSize': tuImgSize,
            'maxObj': maxObj,
            'weightObj': weightObj,
            'weightReg': weightReg,
            'weightCls': weightCls,
            'probThr': probThr,
            'iouThr': iouThr,
            'labelSmoothing': labelSmoothing,
            'batchSize': batchSize,
            'numWorkers': numWorkers,
            'numEpochs': numEpochs,
            'learnRate': learnRate,
            'tuBetas': tuBetas,
            'weightDecay': weightDecay,
            'maxLearnRate': maxLearnRate,
            'seedNum': seedNum,
        },
        'lTrainLoss': lTrainLoss,
        'lTrainScore': lTrainScore,
        'lValLoss': lValLoss,
        'lValScore': lValScore,
        'lLearnRate': lLearnRate,
    })
    torch.save(dCheckPoint, 'BestModel.pt')

# %% Main

if __name__ == '__main__':
    Main(numSamplesTrain, numSamplesVal, T_IMG_SIZE, maxObj, len(L_CLASSES), gridSize,
         weightObj, weightReg, weightCls, probThr, iouThr, labelSmoothing, batchSize,
         numWorkers, numEpochs, learnRate, tuBetas, weightDecay, maxLearnRate, seedNum)