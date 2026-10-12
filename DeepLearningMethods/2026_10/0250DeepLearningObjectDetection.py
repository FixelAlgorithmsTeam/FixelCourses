# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Deep Learning Methods
#
# ## Computer Vision - Object Detection - MNIST Object Detection
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# ## Revision History
#
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 1.0.000 | 12/10/2025 | Royi Avital | First version                                                      |

# %% Packages

import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
from torchvision import tv_tensors
from torch.utils.data import default_collate

import os
import random

from typing import Dict, List, Tuple, Union
from torch import Tensor

# %% Course Packages

from DeepLearningObjectDetection import YoloDataset
from DeepLearningPyTorch import GenDataLoaders, TrainModel

# %% Constants

PROJECT_NAME       = 'FixelCourses'
DATA_FOLDER_NAME   = 'DataSets'
MODELS_FOLDER_NAME = 'Models'
BASE_FOLDER_PATH   = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH   = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)
MODELS_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, MODELS_FOLDER_NAME)

# %% Auxiliary Functions

def CollateObjectDetectionFast( lBatch: List[Tuple[Tensor, Tuple[Tensor, Tensor]]] ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    """Stack equal-sized images and retain variable-length detection targets."""

    tX = default_collate([sample[0] for sample in lBatch]).to(torch.float32)
    lY = [{'Labels': sample[1][0], 'Boxes': sample[1][1]} for sample in lBatch]

    return tX, lY

# %% Parameters

seedNum = 512

# Data
datasetName = 'MNISTObjDet'
dataDirPath = os.path.join(DATA_FOLDER_PATH, datasetName)
tuImgSize   = (128, 128)
imgMode     = torchvision.io.ImageReadMode.GRAY

trainSampleRatio = 0.9
valSampleRatio   = 1 - trainSampleRatio

# Model
modelName   = 'ModelMNISTObjDet_2026_10_12_V2.pt'
gridSize    = 5
numChannels = 1 if imgMode == torchvision.io.ImageReadMode.GRAY else 3

# Training
weightObj      = 1.0
weightReg      = 1.0
weightCls      = 1.0
objPosWeight   = 4.0 #<! Give sparse positive objectness cells more influence than background cells
probThr        = 0.5
iouThr         = 0.5
labelSmoothing = 0.1

batchSize  = 256
numWorkers = 4
numEpochs  = 55

# Optimizer
learnRate    = 1e-5
tuBetas      = (0.9, 0.99)
weightDecay  = 1e-5
maxLearnRate = 2.5e-4

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


class ResidualDepthWiseBlock(nn.Module):
    """Refine features with a cheap residual depthwise-separable convolution."""

    def __init__( self, numChannels: int, dilation: int = 1 ) -> None:
        super().__init__()

        # The depthwise convolution gathers spatial context independently in each channel.
        # Dilation increases the receptive field without reducing the feature-map resolution.
        self.oResidualBranch = nn.Sequential(
            nn.Conv2d(numChannels, numChannels, 3, padding = dilation, dilation = dilation,
                      groups = numChannels, bias = False),
            nn.BatchNorm2d(numChannels),
            nn.ReLU(),
            # The pointwise convolution mixes information between channels.
            nn.Conv2d(numChannels, numChannels, 1, bias = False),
            nn.BatchNorm2d(numChannels),
        )
        self.oActivation = nn.ReLU()

    def forward( self, tX: Tensor ) -> Tensor:

        # The skip connection preserves the original representation and improves gradient flow.
        return self.oActivation(tX + self.oResidualBranch(tX))


class µDetector(nn.Module):
    def __init__( self, numClasses: int, gridSize: int, /, *, numChannels: int = 3 ) -> None:
        super().__init__()

        self.numClasses = numClasses
        self.gridSize   = gridSize

        # Regular convolutions in the backbone provide substantially more spatial and
        # cross-channel capacity than the previous depthwise-only feature extractor.
        # Three stride-2 stages map a 128 x 128 image to a detailed 16 x 16 feature map.
        self.oBackbone = nn.Sequential(
            nn.Conv2d(numChannels, 24, 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 56, 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(56), nn.ReLU(),
            nn.Conv2d(56, 112, 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(112), nn.ReLU(),
        )

        # Residual context blocks preserve the 16 x 16 geometry. Dilations 1, 2 and 4
        # progressively enlarge the receptive field to approximately the full image.
        self.oContext = nn.Sequential(
            ResidualDepthWiseBlock(112, dilation = 1),
            ResidualDepthWiseBlock(112, dilation = 2),
            ResidualDepthWiseBlock(112, dilation = 4),
        )

        # The neck performs one final spatial refinement, compresses the representation,
        # and only then pools it to the requested YOLO grid.
        self.oNeck = nn.Sequential(
            nn.Conv2d(112, 112, 3, padding = 1, groups = 112, bias = False), nn.BatchNorm2d(112), nn.ReLU(),
            nn.Conv2d(112, 80, 1, bias = False), nn.BatchNorm2d(80), nn.ReLU(),
            nn.AdaptiveAvgPool2d((gridSize, gridSize)),
        )

        # Separate heads prevent class prediction from sharing its final projection with
        # objectness and localization. Each head still predicts once per grid cell.
        self.oBoxHead = nn.Conv2d(80, 5, 1) #<! Objectness logit followed by 4 box values
        self.oClsHead = nn.Conv2d(80, numClasses, 1) #<! Class logits

    def forward( self, tX: Tensor ) -> Tensor:

        tX = self.oBackbone(tX)
        tX = self.oContext(tX)
        tX = self.oNeck(tX)

        tBox = self.oBoxHead(tX)
        tCls = self.oClsHead(tX)
        # Keep objectness and classes as logits for their classification losses.
        # Constrain only the cell-relative center and normalized size to [0, 1].
        tBox = torch.cat((tBox[:, :1], torch.sigmoid(tBox[:, 1:])), dim = 1)
        tX = torch.cat((tBox, tCls), dim = 1)

        return tX


class ObjDetLoss(nn.Module):
    def __init__( self, tuImgSize: Tuple[int, int], numCls: int, λObj: float, λReg: float, λCls: float, objPosWeight: float = 4.0, ϵ: float = 0.0 ) -> None:
        super().__init__()

        self.numCls       = numCls
        self.λObj         = λObj
        self.λReg         = λReg
        self.λCls         = λCls
        self.ϵ            = ϵ
        self.tuImgSize    = tuImgSize
        self.boxL1Weight  = 0.25
        # Positive objectness cells are sparse. pos_weight increases their BCE contribution
        # without reducing the penalty for false positives in background cells.
        self.oObjLoss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(objPosWeight))
        # Smooth L1 stabilizes encoded center/size learning while CIoU below directly
        # optimizes overlap, center distance and aspect-ratio agreement.
        self.oRegLoss = nn.SmoothL1Loss()
        self.oClsLoss = nn.CrossEntropyLoss(label_smoothing = ϵ)

    def forward( self, tYHat: Tensor, lY: List[Dict[str, Tensor]] ) -> Tensor:

        batchSize = tYHat.shape[0]
        gridSize  = tYHat.shape[2]
        runDevice = tYHat.device
        numRows, numCols = self.tuImgSize

        tObjTgt = torch.zeros((batchSize, gridSize, gridSize), device = runDevice, dtype = tYHat.dtype)
        tRegTgt = torch.zeros((batchSize, gridSize, gridSize, 4), device = runDevice, dtype = tYHat.dtype)
        tClsTgt = torch.zeros((batchSize, gridSize, gridSize), device = runDevice, dtype = torch.long)

        for sampleIdx in range(batchSize):
            tB = lY[sampleIdx]['Boxes'].to(device = runDevice, dtype = tYHat.dtype)
            tC = lY[sampleIdx]['Labels'].to(device = runDevice, dtype = torch.long)

            if tB.numel() == 0:
                continue

            tB = torchvision.ops.box_convert(tB, in_fmt = 'xyxy', out_fmt = 'cxcywh')
            tB = tB / tB.new_tensor([numCols, numRows, numCols, numRows])
            vCxNorm, vCyNorm, vW, vH = tB.T
            vCellX = (vCxNorm * gridSize).floor().to(torch.long).clamp_(0, gridSize - 1)
            vCellY = (vCyNorm * gridSize).floor().to(torch.long).clamp_(0, gridSize - 1)
            vCxRel = vCxNorm * gridSize - vCellX.to(vCxNorm.dtype)
            vCyRel = vCyNorm * gridSize - vCellY.to(vCyNorm.dtype)

            tObjTgt[sampleIdx, vCellY, vCellX]     = 1.0
            tRegTgt[sampleIdx, vCellY, vCellX, :] = torch.stack((vCxRel, vCyRel, vW, vH), dim = 1)
            tClsTgt[sampleIdx, vCellY, vCellX]     = tC

        objLoss = self.oObjLoss(tYHat[:, 0, :, :], tObjTgt)
        mObjMask = tObjTgt.to(torch.bool)
        if mObjMask.any():
            tYHatReg = tYHat[:, 1:5, :, :].permute(0, 2, 3, 1)
            mBPred = tYHatReg[mObjMask]
            mBTgt  = tRegTgt[mObjMask]

            # Decode positive cells to normalized image XYXY boxes before calculating CIoU.
            # This makes the main regression term match the geometry used at evaluation time.
            _, vCellY, vCellX = torch.where(mObjMask)
            mBPredImg = mBPred.clone()
            mBTgtImg  = mBTgt.clone()
            mBPredImg[:, 0] = (vCellX + mBPredImg[:, 0]) / gridSize
            mBPredImg[:, 1] = (vCellY + mBPredImg[:, 1]) / gridSize
            mBTgtImg[:, 0]  = (vCellX + mBTgtImg[:, 0]) / gridSize
            mBTgtImg[:, 1]  = (vCellY + mBTgtImg[:, 1]) / gridSize
            mBPredImg = torchvision.ops.box_convert(mBPredImg, in_fmt = 'cxcywh', out_fmt = 'xyxy')
            mBTgtImg  = torchvision.ops.box_convert(mBTgtImg, in_fmt = 'cxcywh', out_fmt = 'xyxy')

            ciouLoss = torchvision.ops.complete_box_iou_loss(mBPredImg, mBTgtImg, reduction = 'mean')
            regLoss  = ciouLoss + (self.boxL1Weight * self.oRegLoss(mBPred, mBTgt))
            tYHatCls = tYHat[:, 5:, :, :].permute(0, 2, 3, 1)
            clsLoss  = self.oClsLoss(tYHatCls[mObjMask], tClsTgt[mObjMask])
        else:
            regLoss = tYHat.new_zeros(())
            clsLoss = tYHat.new_zeros(())

        return (self.λObj * objLoss) + (self.λReg * regLoss) + (self.λCls * clsLoss)


class ObjDetScore(nn.Module):
    def __init__( self, tuImgSize: Tuple[int, int], numCls: int, gridSize: int, probThr: float, iouThr: float, runDevice: torch.device ) -> None:
        super().__init__()

        self.numCls    = numCls
        self.gridSize  = gridSize
        self.probThr   = probThr
        self.iouThr    = iouThr
        self.runDevice = runDevice
        self.tuImgSize = tuImgSize

    @torch.inference_mode()
    def forward( self, tYHat: Tensor, lY: List[Dict[str, Tensor]] ) -> float:

        tYHat = tYHat.detach().float().cpu()
        numRows, numCols = self.tuImgSize
        valScore = 0.0

        for sampleIdx, dTarget in enumerate(lY):
            mBGt = dTarget['Boxes'].detach().float().cpu()
            vYGt = dTarget['Labels'].detach().cpu()
            vPy, vPx = torch.where(torch.sigmoid(tYHat[sampleIdx, 0]) >= self.probThr)
            numGt, numPred = len(vYGt), len(vPx)

            if numGt + numPred == 0:
                valScore += 1.0
                continue
            if numGt == 0 or numPred == 0:
                continue

            mBPred = tYHat[sampleIdx, 1:5, vPy, vPx].T.clone()
            mBPred[:, :2] = (mBPred[:, :2] + torch.stack((vPx, vPy), dim = 1)) / self.gridSize
            mBPred = torchvision.ops.box_convert(mBPred, in_fmt = 'cxcywh', out_fmt = 'xyxy')
            mBPred *= mBPred.new_tensor([numCols, numRows, numCols, numRows])
            vYPred = tYHat[sampleIdx, 5:5 + self.numCls, vPy, vPx].argmax(dim = 0)

            mIoU = torchvision.ops.box_iou(mBPred, mBGt)
            mMatch = ((mIoU >= self.iouThr) & (vYPred[:, None] == vYGt[None, :])).numpy()
            vPredIdx, vGtIdx = sp.optimize.linear_sum_assignment(mMatch, maximize = True)
            numTp = int(mMatch[vPredIdx, vGtIdx].sum())
            valScore += 2 * numTp / (numPred + numGt)

        return float(valScore / len(lY)) if lY else 0.0

# %% Main Function

def Main( dataDirPath: str, modelName: str, tuImgSize: Tuple[int, int], imgMode: torchvision.io.ImageReadMode,
          trainSampleRatio: float, valSampleRatio: float, gridSize: int, weightObj: float,
          weightReg: float, weightCls: float, objPosWeight: float, probThr: float, iouThr: float,
          labelSmoothing: float, batchSize: int, numWorkers: int, numEpochs: int,
          learnRate: float, tuBetas: Tuple[float, float], weightDecay: float,
          maxLearnRate: float, seedNum: int ) -> None:

    if not os.path.isdir(dataDirPath):
        raise FileNotFoundError(f'Dataset folder does not exist: {dataDirPath}')
    if batchSize <= 0 or numWorkers < 0 or numEpochs <= 0:
        raise ValueError('batchSize and numEpochs must be positive; numWorkers cannot be negative.')

    np.random.seed(seedNum)
    random.seed(seedNum)
    torch.manual_seed(seedNum)
    torch.backends.cudnn.benchmark = True

    dsData = YoloDataset(dataDirPath, imgMode = imgMode)
    numSamples = len(dsData)
    numCls = dsData.GetNumClasses()
    if numSamples < 2:
        raise ValueError('The dataset must contain at least two samples.')

    tXSample, _ = dsData[0]
    sampleImgSize = tuple(tXSample.shape[-2:])
    if sampleImgSize != tuImgSize:
        raise ValueError(f'tuImgSize {tuImgSize} does not match dataset images {sampleImgSize}.')

    vIdxTrain, vIdxVal = train_test_split(np.arange(numSamples), test_size = valSampleRatio,
                                          train_size = trainSampleRatio, random_state = seedNum,
                                          shuffle = True)
    dsTrain = torch.utils.data.Subset(dsData, vIdxTrain)
    dsVal   = torch.utils.data.Subset(dsData, vIdxVal)

    persistentWorkers = numWorkers > 0
    pinMemory = torch.cuda.is_available()
    dlTrain, dlVal = GenDataLoaders(dsTrain, dsVal, batchSize, numWorkers = numWorkers,
                                   CollateFn = CollateObjectDetectionFast, pinMemory = pinMemory,
                                   dropLast = True, persWork = persistentWorkers)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    numChannels = 1 if imgMode == torchvision.io.ImageReadMode.GRAY else 3
    oModel = µDetector(numCls, gridSize, numChannels = numChannels).to(runDevice)
    hL = ObjDetLoss(tuImgSize, numCls, weightObj, weightReg, weightCls, objPosWeight, labelSmoothing).to(runDevice)
    hS = ObjDetScore(tuImgSize, numCls, gridSize, probThr, iouThr, runDevice).to(runDevice)

    print(f'Running on device: {runDevice}')
    print(f'The training data set contains  : {len(dsTrain):5d} samples.')
    print(f'The validation data set contains: {len(dsVal):5d} samples.')
    print(f'Image spatial size: {tuImgSize}, Grid size: {gridSize}')
    print(f'Model parameters: {sum(parameter.numel() for parameter in oModel.parameters()):,}')
    print(f'Workers per loader: {numWorkers}, Persistent workers: {persistentWorkers}, Pinned memory: {pinMemory}')
    print(f'Prefetch factor: {dlTrain.prefetch_factor}')

    oOpt = torch.optim.AdamW(oModel.parameters(), lr = learnRate, betas = tuBetas, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = maxLearnRate, total_steps = numEpochs)

    _, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch)

    dCheckPoint = torch.load('BestModel.pt', map_location = 'cpu', weights_only = True)
    dCheckPoint.update({
        'ModelParams': {
            'numClasses': numCls,
            'gridSize': gridSize,
            'numChannels': numChannels,
        },
        'TrainParams': {
            'dataDirPath': dataDirPath,
            'tuImgSize': tuImgSize,
            'imgMode': imgMode.name,
            'trainSampleRatio': trainSampleRatio,
            'valSampleRatio': valSampleRatio,
            'weightObj': weightObj,
            'weightReg': weightReg,
            'weightCls': weightCls,
            'objPosWeight': objPosWeight,
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

    # modelPath = os.path.join(MODELS_FOLDER_PATH, modelName)
    # os.makedirs(MODELS_FOLDER_PATH, exist_ok = True)
    torch.save(dCheckPoint, 'BestModel.pt')
    print(f'Model and training history saved to: BestModel.pt')
    # torch.save(dCheckPoint, modelPath)
    # print(f'Model and training history saved to: {modelPath}')

# %% Main

if __name__ == '__main__':
    Main(dataDirPath, modelName, tuImgSize, imgMode, trainSampleRatio, valSampleRatio,
         gridSize, weightObj, weightReg, weightCls, objPosWeight, probThr, iouThr, labelSmoothing,
         batchSize, numWorkers, numEpochs, learnRate, tuBetas, weightDecay,
         maxLearnRate, seedNum)