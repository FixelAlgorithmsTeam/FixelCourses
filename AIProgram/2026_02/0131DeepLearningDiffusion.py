# %% [markdown]
#
# # Deep Learning Methods
#
# ## Deep Learning - Computer Vision - Conditional Diffusion Models
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com

# %% Packages

import os
import random
import time
from typing import Callable, List, Literal, Optional, Tuple
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.regression import r2_score
from torchvision.io import decode_image
from torchvision.transforms import v2 as TorchVisionTrns

# %% Configuration

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

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

# %% Data Set

class SatAerialMapDataset(Dataset):
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
            tX = TorchVisionTrns.functional.resize(tX, size = (self._imgSize, self._imgSize), interpolation = TorchVisionTrns.InterpolationMode.BILINEAR, antialias = True)
            tY = TorchVisionTrns.functional.resize(tY, size = (self._imgSize, self._imgSize), interpolation = TorchVisionTrns.InterpolationMode.BILINEAR, antialias = True)

        if self._hTrns:
            tX = self._hTrns(tX)

        tY = TorchVisionTrns.functional.to_dtype(tY, torch.float, scale = True)

        return tX, tY

# %% Diffusion Schedule

class DiffusionSchedule:
    def __init__( self, numSteps: int, runDevice: torch.device = torch.device('cpu') ) -> None:
        self.numSteps = numSteps
        vGrid = torch.linspace(0, 1, numSteps + 1, dtype = torch.float64)
        vCurve = torch.cos((vGrid + 0.008) / 1.008 * np.pi / 2).square()
        vBeta = (1 - vCurve[1:] / vCurve[:-1]).clamp(1e-5, 0.999)
        vAlpha = 1 - vBeta
        vAlphaBar = torch.cumprod(vAlpha, dim = 0)
        vAlphaPrev = torch.cat((torch.ones(1, dtype = torch.float64), vAlphaBar[:-1]))
        self.vBeta = vBeta.float().to(runDevice)
        self.vAlphaBar = vAlphaBar.float().to(runDevice)
        self.vVariance = (vBeta * (1 - vAlphaPrev) / (1 - vAlphaBar)).float().to(runDevice)
        self.vCoefClean = (vBeta * vAlphaPrev.sqrt() / (1 - vAlphaBar)).float().to(runDevice)
        self.vCoefNoisy = (vAlpha.sqrt() * (1 - vAlphaPrev) / (1 - vAlphaBar)).float().to(runDevice)

    def AddNoise( self, tClean: Tensor, vTime: Tensor, tNoise: Tensor ) -> Tensor:

        tAlphaBar = self.vAlphaBar[vTime].view(-1, 1, 1, 1)

        return tAlphaBar.sqrt() * tClean + (1 - tAlphaBar).sqrt() * tNoise

    def Step( self, tNoisy: Tensor, tNoiseHat: Tensor, stepIdx: int, tNoise: Tensor ) -> Tensor:

        alphaBar = self.vAlphaBar[stepIdx]
        tClean = ((tNoisy - (1 - alphaBar).sqrt() * tNoiseHat) / alphaBar.sqrt()).clamp(-1, 1)
        tMean = self.vCoefClean[stepIdx] * tClean + self.vCoefNoisy[stepIdx] * tNoisy

        return tMean + self.vVariance[stepIdx].sqrt() * tNoise

# %% Conditional Denoiser

class TimeBlock(nn.Module):
    def __init__( self, inCh: int, outCh: int, timeDim: int, *, useSeparable: bool = True ) -> None:
        super().__init__()

        oConv = nn.Conv2d(inCh, outCh, 3, padding = 1)
        if useSeparable and min(inCh, outCh) >= 64:
            oConv = nn.Sequential(nn.Conv2d(inCh, inCh, 3, padding = 1, groups = inCh, bias = False), nn.Conv2d(inCh, outCh, 1))
        oOut = nn.Conv2d(outCh, outCh, 3, padding = 1)
        if useSeparable and outCh >= 64:
            oOut = nn.Sequential(nn.Conv2d(outCh, outCh, 3, padding = 1, groups = outCh, bias = False), nn.Conv2d(outCh, outCh, 1))

        self.oConv = nn.Sequential(oConv, nn.GroupNorm(8, outCh), nn.SiLU())
        self.oTime = nn.Linear(timeDim, outCh)
        self.oOut = nn.Sequential(nn.GroupNorm(8, outCh), nn.SiLU(), oOut)
        self.oSkip = nn.Conv2d(inCh, outCh, 1) if inCh != outCh else nn.Identity()

    def forward( self, tX: Tensor, mTime: Tensor ) -> Tensor:

        tZ = self.oConv(tX) + self.oTime(mTime)[:, :, None, None]

        return self.oOut(tZ) + self.oSkip(tX)

class AttentionBlock(nn.Module):
    def __init__( self, numCh: int, numHeads: int = 4 ) -> None:
        super().__init__()

        self.numHeads = numHeads
        self.oNorm = nn.GroupNorm(8, numCh)
        self.oQKV = nn.Conv2d(numCh, 3 * numCh, 1)
        self.oOut = nn.Conv2d(numCh, numCh, 1)
        nn.init.zeros_(self.oOut.weight)
        nn.init.zeros_(self.oOut.bias)

    def forward( self, tX: Tensor ) -> Tensor:

        numB, numCh, numRows, numCols = tX.shape
        tQ, tK, tV = self.oQKV(self.oNorm(tX)).view(numB, 3, self.numHeads, numCh // self.numHeads, numRows * numCols).transpose(-1, -2).unbind(1)
        tZ = F.scaled_dot_product_attention(tQ, tK, tV)

        return tX + self.oOut(tZ.transpose(-1, -2).reshape(numB, numCh, numRows, numCols))

class ConditionalUNet(nn.Module):
    def __init__( self, baseCh: int = 32, timeDim: int = 128, *, useSeparable: bool = False ) -> None:
        super().__init__()

        self.vFreq = torch.exp(-np.log(10000.0) * torch.arange(timeDim // 2) / (timeDim // 2 - 1))
        self.oTime = nn.Sequential(nn.Linear(timeDim, timeDim), nn.SiLU(), nn.Linear(timeDim, timeDim))
        self.oEnc1 = TimeBlock(7, baseCh, timeDim, useSeparable = useSeparable)
        self.oEnc2 = TimeBlock(baseCh, 2 * baseCh, timeDim, useSeparable = useSeparable)
        self.oEnc3 = TimeBlock(2 * baseCh, 4 * baseCh, timeDim, useSeparable = useSeparable)
        self.oEnc4 = TimeBlock(4 * baseCh, 8 * baseCh, timeDim, useSeparable = useSeparable)
        self.oMid = TimeBlock(8 * baseCh, 8 * baseCh, timeDim, useSeparable = useSeparable)
        self.oAttn = AttentionBlock(8 * baseCh)
        self.oUpsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.oDec4 = TimeBlock(16 * baseCh, 8 * baseCh, timeDim, useSeparable = useSeparable)
        self.oDec3 = TimeBlock(12 * baseCh, 4 * baseCh, timeDim, useSeparable = useSeparable)
        self.oDec2 = TimeBlock(6 * baseCh, 2 * baseCh, timeDim, useSeparable = useSeparable)
        self.oDec1 = TimeBlock(3 * baseCh, baseCh, timeDim, useSeparable = useSeparable)
        self.oOut = nn.Conv2d(baseCh, 3, 1)

    def forward( self, tNoisy: Tensor, vTime: Tensor, tSource: Tensor, vCondition: Tensor ) -> Tensor:

        self.vFreq = self.vFreq.to(tNoisy.device)
        mAngles = vTime.float()[:, None] * self.vFreq[None, :]
        mTime = self.oTime(torch.cat((mAngles.sin(), mAngles.cos()), dim = 1))
        tPresent = vCondition.to(tNoisy.dtype).view(-1, 1, 1, 1)
        tMask = tPresent.expand(-1, 1, *tNoisy.shape[-2:])
        tInput = torch.cat((tNoisy, tSource * tPresent, tMask), dim = 1)
        tEnc1 = self.oEnc1(tInput, mTime)
        tEnc2 = self.oEnc2(F.avg_pool2d(tEnc1, 2), mTime)
        tEnc3 = self.oEnc3(F.avg_pool2d(tEnc2, 2), mTime)
        tEnc4 = self.oEnc4(F.avg_pool2d(tEnc3, 2), mTime)
        tZ = self.oAttn(self.oMid(F.avg_pool2d(tEnc4, 2), mTime))
        for tSkip, oBlock in [(tEnc4, self.oDec4), (tEnc3, self.oDec3), (tEnc2, self.oDec2), (tEnc1, self.oDec1)]:
            tZ = self.oUpsample(tZ)
            tZ = oBlock(torch.cat((tZ, tSkip), dim = 1), mTime)

        return self.oOut(tZ)

# %% Loss and Score

def SSIMScore( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return structural_similarity_index_measure(tYHat, tY, data_range = 1.0)

def ImageR2Score( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return r2_score(tYHat.flatten(), tY.flatten(), multioutput = 'uniform_average')

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

# %% Sampling

def PredictGuidedNoise( oModel: nn.Module, tNoisy: Tensor, vTime: Tensor, tSource: Tensor, guidanceScale: float ) -> Tensor:

    vPresent = torch.ones(len(tNoisy), device = tNoisy.device)
    if guidanceScale == 1:
        return oModel(tNoisy, vTime, tSource, vPresent).float()
    tNoiseNull = oModel(tNoisy, vTime, tSource, torch.zeros_like(vPresent)).float()
    if guidanceScale == 0:
        return tNoiseNull
    tNoiseCond = oModel(tNoisy, vTime, tSource, vPresent).float()

    return tNoiseNull + guidanceScale * (tNoiseCond - tNoiseNull)

@torch.inference_mode()
def SampleMaps( oModel: nn.Module, oDiff: DiffusionSchedule, tSource: Tensor, runDevice: torch.device, *, guidanceScale: float = 2.0, sampleSeed: int = 512, numFrames: int = 6 ) -> Tuple[Tensor, list, list]:
    """Generate maps from BCHW aerial images in [0, 1]; numFrames = 0 disables snapshots."""
    oModel.eval()
    tSource = tSource.to(runDevice, non_blocking = True) * 2 - 1
    oGen = torch.Generator(device = runDevice).manual_seed(sampleSeed)
    tNoisy = torch.randn(tSource.shape, device = runDevice, generator = oGen)
    lFrames, lSteps = [], []
    if numFrames > 0:
        lFrames.append(((tNoisy[:1].cpu() + 1) / 2).clamp(0, 1))
        lSteps.append(oDiff.numSteps)
    vSaveSteps = np.linspace(oDiff.numSteps, 0, numFrames, dtype = int)
    for stepIdx in reversed(range(oDiff.numSteps)):
        vTime = torch.full((len(tSource),), stepIdx, device = runDevice, dtype = torch.long)
        with torch.autocast(device_type = runDevice.type, enabled = runDevice.type == 'cuda'):
            tNoiseHat = PredictGuidedNoise(oModel, tNoisy, vTime, tSource, guidanceScale)
        tNoise = torch.randn(tNoisy.shape, device = runDevice, generator = oGen) if stepIdx > 0 else torch.zeros_like(tNoisy)
        tNoisy = oDiff.Step(tNoisy, tNoiseHat, stepIdx, tNoise)
        if stepIdx in vSaveSteps:
            lFrames.append(((tNoisy[:1].cpu() + 1) / 2).clamp(0, 1))
            lSteps.append(stepIdx)

    return ((tNoisy.cpu() + 1) / 2).clamp(0, 1), lFrames, lSteps

# %% Training

def RunDiffusionEpoch( oModel: nn.Module, oDiff: DiffusionSchedule, dlData, hL: Callable, oOpt, *, oScaler = None, dropProb: float = 0.1 ) -> float:

    epochLoss = 0.0
    numSamples = 0
    numBatches = len(dlData)
    runDevice = next(oModel.parameters()).device
    oModel.train(True)

    for batchIdx, (tX, tY) in enumerate(dlData):
        tX = tX.to(runDevice, non_blocking = True) * 2 - 1
        tY = tY.to(runDevice, non_blocking = True) * 2 - 1
        batchSize = tY.shape[0]
        vTime = torch.randint(oDiff.numSteps, (batchSize,), device = runDevice)
        tNoise = torch.randn(tY.shape, device = runDevice)
        tNoisy = oDiff.AddNoise(tY, vTime, tNoise)
        vCondition = (torch.rand(batchSize, device = runDevice) >= dropProb).float()

        with torch.autocast(device_type = runDevice.type, enabled = runDevice.type == 'cuda'):
            tNoiseHat = oModel(tNoisy, vTime, tX, vCondition)
        valLoss = hL(tNoiseHat.float(), tNoise)
        oOpt.zero_grad()
        if oScaler is not None:
            oScaler.scale(valLoss).backward()
            oScaler.unscale_(oOpt)
            nn.utils.clip_grad_norm_(oModel.parameters(), 1.0)
            oScaler.step(oOpt)
            oScaler.update()
        else:
            valLoss.backward()
            nn.utils.clip_grad_norm_(oModel.parameters(), 1.0)
            oOpt.step()

        epochLoss += batchSize * valLoss.detach().item()
        numSamples += batchSize
        print(f'\rTrain Loss - Iteration: {(batchIdx + 1):3d} / {numBatches}, Loss: {valLoss:.6f}', end = '')

    print('', end = '\r')

    return epochLoss / numSamples

@torch.inference_mode()
def EvaluateDiffusionModel( oModel: nn.Module, oDiff: DiffusionSchedule, dlData, hS: Callable, *, guidanceScale: float = 2.0, sampleSeed: int = 512 ) -> float:

    runDevice = next(oModel.parameters()).device
    lGenerated, lTarget = [], []
    for batchIdx, (tX, tY) in enumerate(dlData):
        tGenerated, _, _ = SampleMaps(oModel, oDiff, tX, runDevice, guidanceScale = guidanceScale, sampleSeed = sampleSeed + batchIdx, numFrames = 0)
        lGenerated.append(tGenerated)
        lTarget.append(tY.cpu())
        print(f'\rVal Map Score - Batch: {batchIdx + 1:3d} / {len(dlData)}', end = '')
    print('', end = '\r')

    return hS(torch.cat(lGenerated), torch.cat(lTarget)).item()

def TrainDiffusionModel( oModel: nn.Module, oDiff: DiffusionSchedule, dlTrain, dlVal, oOpt, numEpoch: int, hL: Callable, hS: Callable, *, oSch = None, oScaler = None, dropProb: float = 0.1, guidanceScale: float = 2.0, valEvery: int = 5, sampleSeed: int = 512 ) -> Tuple[nn.Module, List[float], List[int], List[float], List[float]]:

    if valEvery < 1:
        raise ValueError('valEvery must be positive')
    lTrainLoss, lLearnRate = [], []
    lValEpoch, lValScore = [], []
    bestScore = -float('inf')

    for epochIdx in range(numEpoch):
        startTime = time.time()
        learnRate = oOpt.param_groups[0]['lr']
        trainLoss = RunDiffusionEpoch(oModel, oDiff, dlTrain, hL, oOpt, oScaler = oScaler, dropProb = dropProb)
        scoreEpoch = (epochIdx + 1) % valEvery == 0
        if scoreEpoch:
            valScr = EvaluateDiffusionModel(oModel, oDiff, dlVal, hS, guidanceScale = guidanceScale, sampleSeed = sampleSeed)
            lValEpoch.append(epochIdx + 1)
            lValScore.append(valScr)
        if oSch is not None:
            oSch.step()
        epochTime = time.time() - startTime

        lTrainLoss.append(trainLoss)
        lLearnRate.append(learnRate)
        print(f'Epoch {(epochIdx + 1):4d} / {numEpoch}', end = '')
        print(f' | Train Noise Loss: {trainLoss:6.3f}', end = '')
        if scoreEpoch:
            print(f' | Val Score: {valScr:6.3f}', end = '')
        print(f' | Epoch Time: {epochTime:5.2f}', end = '')

        if scoreEpoch and valScr > bestScore:
            bestScore = valScr
            try:
                dCheckPoint = {'Model': oModel.state_dict(), 'Optimizer': oOpt.state_dict()}
                if oSch is not None:
                    dCheckPoint['Scheduler'] = oSch.state_dict()
                torch.save(dCheckPoint, 'BestModel.pt')
                print(' | <-- Checkpoint!', end = '')
            except OSError as oError:
                print(f' | <-- Failed: {oError}', end = '')
        print(' |')

    return oModel, lTrainLoss, lValEpoch, lValScore, lLearnRate

# %% Parameters

# Data
dataSet    = 'SatAerialToMap'
dataSetUrl = r'https://huggingface.co/datasets/Royi/DataSets/resolve/main/SatAerialToMap.zip'
imgSize = 256
trainNumSamples = None
valNumSamples = 16

# Model
baseCh = 32
useSeparable = False
numDiffSteps = 200
conditionDropProb = 0.1
guidanceScale = 2.0

# Training
batchSize = 8
numWorkers = 4
numEpochs = 200
scoreType = 'R2'
valEvery = 10

# Optimizer
ηOpt = 1e-4
tuβ = (0.9, 0.99)
weightDecay = 5e-5
ηSch = 2e-4

# %% Main Function

def Main(
    dataSet: str,
    dataSetUrl: str,
    imgSize: int,
    trainNumSamples: Optional[int],
    valNumSamples: int,
    baseCh: int,
    useSeparable: bool,
    numDiffSteps: int,
    conditionDropProb: float,
    guidanceScale: float,
    batchSize: int,
    numWorkers: int,
    numEpochs: int,
    scoreType: Literal['SSIM', 'R2'],
    valEvery: int,
    ηOpt: float,
    tuβ: Tuple[float, float],
    weightDecay: float,
    ηSch: float,
) -> None:

    datasetFolderPath = os.path.join(DATA_FOLDER_PATH, dataSet)
    if not os.path.isdir(datasetFolderPath):
        fileName = os.path.join(DATA_FOLDER_PATH, f'{dataSet}.zip')
        DownloadUrl(dataSetUrl, fileName)
        with ZipFile(fileName, 'r') as zipFile:
            zipFile.extractall(DATA_FOLDER_PATH)
        time.sleep(1.0)
        os.remove(fileName)

    oTrnsTrain = TorchVisionTrns.Compose([
        TorchVisionTrns.ToDtype(torch.float32, scale = True),
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
    oTrnsVal = TorchVisionTrns.ToDtype(torch.float32, scale = True)

    dsData = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize)
    numSamples = len(dsData)
    dsTrain = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize, hTrns = oTrnsTrain)
    dsVal = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize, hTrns = oTrnsVal)
    vIdxTrain, vIdxVal = train_test_split(np.arange(numSamples), test_size = valNumSamples, train_size = trainNumSamples, random_state = seedNum, shuffle = True)
    dsTrain = torch.utils.data.Subset(dsTrain, vIdxTrain)
    dsVal = torch.utils.data.Subset(dsVal, vIdxVal)

    pinMemory = torch.cuda.is_available()
    persWork = numWorkers > 0
    prefetchFactor = 2 if persWork else None
    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = batchSize, num_workers = numWorkers, pin_memory = pinMemory, drop_last = True, persistent_workers = persWork, prefetch_factor = prefetchFactor)
    dlVal = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = batchSize, num_workers = numWorkers, pin_memory = pinMemory, drop_last = False, persistent_workers = persWork, prefetch_factor = prefetchFactor)

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The training data set contains  : {len(dsTrain):4d} samples.')
    print(f'The validation data set contains: {len(dsVal):4d} samples.')
    print(f'Running on device: {runDevice}')

    oModel = ConditionalUNet(baseCh, useSeparable = useSeparable).to(runDevice)
    oDiff = DiffusionSchedule(numDiffSteps, runDevice)
    hL = nn.MSELoss().to(runDevice)
    hS = Pix2PixScore(scoreType = scoreType).to(runDevice)
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = ηOpt, betas = tuβ, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = ηSch, total_steps = numEpochs, pct_start = 0.1, div_factor = 10, final_div_factor = 20)
    oScaler = torch.amp.GradScaler('cuda', enabled = runDevice.type == 'cuda')

    TrainDiffusionModel(oModel, oDiff, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch, oScaler = oScaler, dropProb = conditionDropProb, guidanceScale = guidanceScale, valEvery = valEvery, sampleSeed = seedNum)

# %% Main

if __name__ == '__main__':
    Main(dataSet, dataSetUrl, imgSize, trainNumSamples, valNumSamples,
         baseCh, useSeparable, numDiffSteps, conditionDropProb, guidanceScale,
         batchSize, numWorkers, numEpochs, scoreType, valEvery,
         ηOpt, tuβ, weightDecay, ηSch)