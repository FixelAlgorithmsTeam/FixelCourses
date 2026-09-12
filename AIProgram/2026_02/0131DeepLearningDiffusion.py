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
import copy
import json
import math
import random
import time
from typing import Callable, Dict, List, Literal, Optional, Tuple
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure, structural_similarity_index_measure
from torchmetrics.functional.regression import r2_score
from torchvision.io import decode_image
from torchvision.transforms import v2 as TorchVisionTrns
from torchvision.utils import make_grid, save_image

# %% Configuration

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)
torch.manual_seed(seedNum) #<! Model init, data order and training noise; `cudnn.benchmark` still adds small run to run differences

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
    def __init__( self, rootFolderPath: str, dataSet: Literal['Train', 'Validation', 'All'], /, *, imgSize: Optional[int] = None, hTrns: Optional[Callable] = None, geoAug: bool = False ) -> None:
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
        self._geoAug    = geoAug

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

        if self._geoAug:
            # Same flip / rotation on both images keeps the pair aligned (covers all 8 symmetries of the square)
            if random.random() < 0.5:
                tX = tX.flip(-1)
                tY = tY.flip(-1)
            numRot = random.randrange(4)
            tX = torch.rot90(tX, numRot, dims = (-2, -1))
            tY = torch.rot90(tY, numRot, dims = (-2, -1))

        if self._hTrns:
            tX = self._hTrns(tX)

        tY = TorchVisionTrns.functional.to_dtype(tY, torch.float, scale = True)

        return tX, tY

def ComputeMapStats( dlData ) -> Tuple[Tuple[float, float, float], float]:
    # One pass over the training loader: per channel mean and a single (global) std of the target maps in [0, 1]

    vSum = torch.zeros(3, dtype = torch.float64)
    sumSq = 0.0
    numPix = 0
    for _, tY in dlData:
        vSum += tY.sum(dim = (0, 2, 3)).double()
        sumSq += tY.double().square().sum().item()
        numPix += tY.shape[0] * tY.shape[2] * tY.shape[3]

    vMean = vSum / numPix
    mapStd = math.sqrt(sumSq / (3 * numPix) - vMean.mean().item() ** 2)

    return tuple(vMean.tolist()), float(mapStd)

# %% Diffusion Schedule

class DiffusionSchedule:
    def __init__( self, numSteps: int, runDevice: torch.device = torch.device('cpu'), *, vMean: Tuple[float, float, float] = (0.5, 0.5, 0.5), dataStd: float = 0.5, predictType: Literal['Noise', 'Clean'] = 'Noise' ) -> None:
        # Diffusion runs on the normalized target (y - vMean) / dataStd so the data is zero mean like the noise prior; defaults reproduce 2y - 1
        # predictType: what the network outputs, the added noise ('Noise') or the clean map ('Clean'); both give the clean estimate used by `Step`
        if predictType not in ('Noise', 'Clean'):
            raise ValueError("predictType must be 'Noise' or 'Clean'")
        self.numSteps = numSteps
        self.predictType = predictType
        vGrid = torch.linspace(0, 1, numSteps + 1, dtype = torch.float64)
        vCurve = torch.cos((vGrid + 0.008) / 1.008 * math.pi / 2).square()
        vBeta = (1 - vCurve[1:] / vCurve[:-1]).clamp(1e-5, 0.999)
        vAlpha = 1 - vBeta
        vAlphaBar = torch.cumprod(vAlpha, dim = 0)
        vAlphaPrev = torch.cat((torch.ones(1, dtype = torch.float64), vAlphaBar[:-1]))
        self.vBeta = vBeta.float().to(runDevice)
        self.vAlphaBar = vAlphaBar.float().to(runDevice)
        self.vVariance = (vBeta * (1 - vAlphaPrev) / (1 - vAlphaBar)).float().to(runDevice)
        self.vCoefClean = (vBeta * vAlphaPrev.sqrt() / (1 - vAlphaBar)).float().to(runDevice)
        self.vCoefNoisy = (vAlpha.sqrt() * (1 - vAlphaPrev) / (1 - vAlphaBar)).float().to(runDevice)
        self.tMean = torch.tensor(vMean).view(1, -1, 1, 1).to(runDevice)
        self.dataStd = float(dataStd) #<! Plain float so the checkpoint stays loadable with `weights_only = True`
        self.tClampMin = self.Normalize(torch.zeros_like(self.tMean)) #<! Image value 0 in diffusion space
        self.tClampMax = self.Normalize(torch.ones_like(self.tMean)) #<! Image value 1 in diffusion space

    def Normalize( self, tImg: Tensor ) -> Tensor:
        # Image in [0, 1] -> diffusion space

        return (tImg - self.tMean) / self.dataStd

    def Denormalize( self, tClean: Tensor ) -> Tensor:
        # Diffusion space -> image in [0, 1]

        return (tClean * self.dataStd + self.tMean).clamp(0, 1)

    def AddNoise( self, tClean: Tensor, vTime: Tensor, tNoise: Tensor ) -> Tensor:

        tAlphaBar = self.vAlphaBar[vTime].view(-1, 1, 1, 1)

        return tAlphaBar.sqrt() * tClean + (1 - tAlphaBar).sqrt() * tNoise

    def Target( self, tClean: Tensor, tNoise: Tensor ) -> Tensor:
        # The regression target of the network

        return tNoise if self.predictType == 'Noise' else tClean

    def PredictClean( self, tNoisy: Tensor, tPred: Tensor, vTime ) -> Tensor:
        # Clean map estimate from the network output (`vTime` is a batch of indices or a single index)

        if self.predictType == 'Clean':
            return tPred
        tAlphaBar = self.vAlphaBar[vTime].view(-1, 1, 1, 1)

        return (tNoisy - (1 - tAlphaBar).sqrt() * tPred) / tAlphaBar.sqrt()

    def Step( self, tNoisy: Tensor, tPred: Tensor, stepIdx: int, tNoise: Tensor ) -> Tensor:

        tClean = self.PredictClean(tNoisy, tPred, stepIdx).clamp(self.tClampMin, self.tClampMax)
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

class TimeStage(nn.Module):
    def __init__( self, inCh: int, outCh: int, timeDim: int, numBlocks: int, *, useSeparable: bool = True ) -> None:
        super().__init__()

        self.lBlocks = nn.ModuleList([TimeBlock(inCh if ii == 0 else outCh, outCh, timeDim, useSeparable = useSeparable) for ii in range(numBlocks)])

    def forward( self, tX: Tensor, mTime: Tensor ) -> Tensor:

        for oBlock in self.lBlocks:
            tX = oBlock(tX, mTime)

        return tX

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
    def __init__( self, baseCh: int = 32, timeDim: int = 128, numBlocks: int = 2, *, lChMult: Tuple[int, int, int, int] = (1, 2, 4, 8), useSeparable: bool = False ) -> None:
        # lChMult: channel multiplier per level (full resolution -> 1/8); every baseCh * mult must be a multiple of 8 (GroupNorm) and the last two of 4 (attention heads)
        super().__init__()

        ch1, ch2, ch3, ch4 = (baseCh * chMult for chMult in lChMult)
        self.vFreq = torch.exp(-math.log(10000.0) * torch.arange(timeDim // 2) / (timeDim // 2 - 1))
        self.oTime = nn.Sequential(nn.Linear(timeDim, timeDim), nn.SiLU(), nn.Linear(timeDim, timeDim))
        self.oEnc1 = TimeStage(7, ch1, timeDim, numBlocks, useSeparable = useSeparable)
        self.oEnc2 = TimeStage(ch1, ch2, timeDim, numBlocks, useSeparable = useSeparable)
        self.oEnc3 = TimeStage(ch2, ch3, timeDim, numBlocks, useSeparable = useSeparable)
        self.oAttnEnc3 = AttentionBlock(ch3) #<! 1/8 resolution (32 x 32 for a 256 input): region consistency, ~16x the bottleneck attention cost
        self.oEnc4 = TimeStage(ch3, ch4, timeDim, numBlocks, useSeparable = useSeparable)
        self.oMid = TimeBlock(ch4, ch4, timeDim, useSeparable = useSeparable)
        self.oAttn = AttentionBlock(ch4)
        self.oUpsample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.oDec4 = TimeStage(2 * ch4, ch4, timeDim, numBlocks, useSeparable = useSeparable)
        self.oDec3 = TimeStage(ch4 + ch3, ch3, timeDim, numBlocks, useSeparable = useSeparable)
        self.oAttnDec3 = AttentionBlock(ch3)
        self.oDec2 = TimeStage(ch3 + ch2, ch2, timeDim, numBlocks, useSeparable = useSeparable)
        self.oDec1 = TimeStage(ch2 + ch1, ch1, timeDim, numBlocks, useSeparable = useSeparable)
        self.oOut = nn.Conv2d(ch1, 3, 1)

    def forward( self, tNoisy: Tensor, vTime: Tensor, tSource: Tensor, vCondition: Tensor ) -> Tensor:

        self.vFreq = self.vFreq.to(tNoisy.device)
        mAngles = vTime.float()[:, None] * self.vFreq[None, :]
        mTime = self.oTime(torch.cat((mAngles.sin(), mAngles.cos()), dim = 1))
        tPresent = vCondition.to(tNoisy.dtype).view(-1, 1, 1, 1)
        tMask = tPresent.expand(-1, 1, *tNoisy.shape[-2:])
        tInput = torch.cat((tNoisy, tSource * tPresent, tMask), dim = 1)
        tEnc1 = self.oEnc1(tInput, mTime)
        tEnc2 = self.oEnc2(F.avg_pool2d(tEnc1, 2), mTime)
        tEnc3 = self.oAttnEnc3(self.oEnc3(F.avg_pool2d(tEnc2, 2), mTime))
        tEnc4 = self.oEnc4(F.avg_pool2d(tEnc3, 2), mTime)
        tZ = self.oAttn(self.oMid(F.avg_pool2d(tEnc4, 2), mTime))
        tZ = self.oDec4(torch.cat((self.oUpsample(tZ), tEnc4), dim = 1), mTime)
        tZ = self.oAttnDec3(self.oDec3(torch.cat((self.oUpsample(tZ), tEnc3), dim = 1), mTime))
        tZ = self.oDec2(torch.cat((self.oUpsample(tZ), tEnc2), dim = 1), mTime)
        tZ = self.oDec1(torch.cat((self.oUpsample(tZ), tEnc1), dim = 1), mTime)

        return self.oOut(tZ)

# %% Loss and Score

class Pix2PixLoss(nn.Module):
    def __init__( self, lossType: Literal['L1', 'SmoothL1', 'L2', 'MSE'] = 'MSE' ) -> None:
        # Applied to the network output and its target (noise or clean map, per `predictType`)
        super().__init__()

        match lossType:
            case 'L1':
                self.oLoss = nn.L1Loss()
            case 'SmoothL1':
                self.oLoss = nn.SmoothL1Loss()
            case 'L2' | 'MSE':
                self.oLoss = nn.MSELoss()
            case _:
                raise ValueError('The parameter `lossType` must be either `L1`, `SmoothL1`, `L2` or `MSE`')

    def forward( self, tYHat: Tensor, tY: Tensor ) -> Tensor:

        return self.oLoss(tYHat, tY)

def SSIMScore( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return structural_similarity_index_measure(tYHat, tY, data_range = 1.0)

def MSSSIMScore( tYHat: Tensor, tY: Tensor ) -> Tensor:
    # 5 scales with weights (0.045, 0.286, 0.300, 0.236, 0.133): sees region level color / content and tolerates ~1 px shifts, but nearly ignores fine blur / speckle

    return multiscale_structural_similarity_index_measure(tYHat, tY, data_range = 1.0)

def ImageR2Score( tYHat: Tensor, tY: Tensor ) -> Tensor:

    return r2_score(tYHat.flatten(), tY.flatten(), multioutput = 'uniform_average')

def TotalVariation( tImg: Tensor ) -> Tensor:
    # Mean anisotropic TV per image (B x C x H x W -> B): low for piecewise constant images, high for blur free noise / speckle

    tDiffRow = (tImg[:, :, 1:, :] - tImg[:, :, :-1, :]).abs().mean(dim = (1, 2, 3))
    tDiffCol = (tImg[:, :, :, 1:] - tImg[:, :, :, :-1]).abs().mean(dim = (1, 2, 3))

    return tDiffRow + tDiffCol

def TVAgreementScore( tYHat: Tensor, tY: Tensor ) -> Tensor:
    # min(r, 1 / r) with r = TV(yHat) / TV(y), in [0, 1]: symmetric in log scale, so "twice the edges" (speckle) and "half the edges" (blur) score the same 0.5
    # A flat image (r = 0) scores 0, matching the reference's amount of edges scores 1

    vTVHat, vTV = TotalVariation(tYHat), TotalVariation(tY)
    vRatio      = vTVHat / vTV.clamp(min = 1e-6)

    return torch.minimum(vRatio, 1 / vRatio.clamp(min = 1e-6)).mean()

def MapTVScore( tYHat: Tensor, tY: Tensor ) -> Tensor:
    # Content (R2), structure (SSIM) and piecewise constant agreement (TV); R2 anchors the score to the right map, TV penalizes blur / speckle

    return (ImageR2Score(tYHat, tY).clamp(0, 1) + SSIMScore(tYHat, tY) + TVAgreementScore(tYHat, tY)) / 3

def MapTVMSScore( tYHat: Tensor, tY: Tensor ) -> Tensor:
    # As `MapTVScore` with MS-SSIM: the multi scale term covers content / color / small shifts, the TV term covers the sharpness MS-SSIM ignores

    return (ImageR2Score(tYHat, tY).clamp(0, 1) + MSSSIMScore(tYHat, tY) + TVAgreementScore(tYHat, tY)) / 3

class Pix2PixScore(nn.Module):
    def __init__( self, scoreType: Literal['SSIM', 'MSSSIM', 'R2', 'MapTV', 'MapTVMS'] = 'SSIM' ) -> None:
        super().__init__()

        match scoreType:
            case 'SSIM':
                self.hScore = SSIMScore
            case 'MSSSIM':
                self.hScore = MSSSIMScore
            case 'R2':
                self.hScore = ImageR2Score
            case 'MapTV':
                self.hScore = MapTVScore
            case 'MapTVMS':
                self.hScore = MapTVMSScore
            case _:
                raise ValueError('The parameter `scoreType` must be either `SSIM`, `MSSSIM`, `R2`, `MapTV` or `MapTVMS`')

    def forward( self, tYHat: Tensor, tY: Tensor ) -> Tensor:

        return self.hScore(tYHat, tY)

# %% Sampling

def PredictGuided( oModel: nn.Module, tNoisy: Tensor, vTime: Tensor, tSource: Tensor, guidanceScale: float ) -> Tensor:
    # Classifier free guidance on the network output (valid for both noise and clean prediction, the combination is linear)

    vPresent = torch.ones(len(tNoisy), device = tNoisy.device)
    if guidanceScale == 1:
        return oModel(tNoisy, vTime, tSource, vPresent).float()
    tPredNull = oModel(tNoisy, vTime, tSource, torch.zeros_like(vPresent)).float()
    if guidanceScale == 0:
        return tPredNull
    tPredCond = oModel(tNoisy, vTime, tSource, vPresent).float()

    return tPredNull + guidanceScale * (tPredCond - tPredNull)

@torch.inference_mode()
def SampleMaps( oModel: nn.Module, oDiff: DiffusionSchedule, tSource: Tensor, runDevice: torch.device, *, guidanceScale: float = 2.0, sampleSeed: int = 512, numFrames: int = 6 ) -> Tuple[Tensor, list, list]:
    """Generate maps from BCHW aerial images in [0, 1]; numFrames = 0 disables snapshots."""
    oModel.eval()
    tSource = tSource.to(runDevice, non_blocking = True) * 2 - 1
    oGen = torch.Generator(device = runDevice).manual_seed(sampleSeed)
    tNoisy = torch.randn(tSource.shape, device = runDevice, generator = oGen)
    lFrames, lSteps = [], []
    if numFrames > 0:
        lFrames.append(oDiff.Denormalize(tNoisy[:1]).cpu())
        lSteps.append(oDiff.numSteps)
    vSaveSteps = np.linspace(oDiff.numSteps, 0, numFrames, dtype = int)
    for stepIdx in reversed(range(oDiff.numSteps)):
        vTime = torch.full((len(tSource),), stepIdx, device = runDevice, dtype = torch.long)
        with torch.autocast(device_type = runDevice.type, enabled = runDevice.type == 'cuda'):
            tPred = PredictGuided(oModel, tNoisy, vTime, tSource, guidanceScale)
        tNoise = torch.randn(tNoisy.shape, device = runDevice, generator = oGen) if stepIdx > 0 else torch.zeros_like(tNoisy)
        tNoisy = oDiff.Step(tNoisy, tPred, stepIdx, tNoise)
        if stepIdx in vSaveSteps:
            lFrames.append(oDiff.Denormalize(tNoisy[:1]).cpu())
            lSteps.append(stepIdx)

    return oDiff.Denormalize(tNoisy).cpu(), lFrames, lSteps

# %% Training

def WarmupHoldCosine( numEpochs: int, warmupFrac: float, holdFrac: float, minRatio: float ) -> Callable[[int], float]:
    # LR multiplier per epoch: linear warmup -> hold at the peak -> cosine decay to `minRatio` of the peak (never reaches zero)

    numWarmup = max(1, round(warmupFrac * numEpochs))
    numHold   = round(holdFrac * numEpochs)
    numDecay  = max(1, numEpochs - numWarmup - numHold)

    def hLrRatio( epochIdx: int ) -> float:
        if epochIdx < numWarmup:
            return (epochIdx + 1) / numWarmup
        if epochIdx < numWarmup + numHold:
            return 1.0
        decayFrac = min(1.0, (epochIdx - numWarmup - numHold) / numDecay)
        return minRatio + (1.0 - minRatio) * 0.5 * (1.0 + math.cos(math.pi * decayFrac))

    return hLrRatio

class ModelEma:
    def __init__( self, oModel: nn.Module, decay: float = 0.9995 ) -> None:
        # Exponential moving average of the weights; the decay ramps up from 0.1 so the first updates do not anchor the average to the random init

        self.oModel = copy.deepcopy(oModel).eval()
        for tParam in self.oModel.parameters():
            tParam.requires_grad_(False)
        self.decay = decay
        self.numUpdates = 0

    @torch.no_grad()
    def Update( self, oModel: nn.Module ) -> None:

        self.numUpdates += 1
        decay = min(self.decay, (1 + self.numUpdates) / (10 + self.numUpdates))
        lEma = list(self.oModel.parameters())
        lParam = [tParam.detach() for tParam in oModel.parameters()]
        torch._foreach_lerp_(lEma, lParam, 1 - decay) #<! ema = decay * ema + (1 - decay) * param

def RunDiffusionEpoch( oModel: nn.Module, oDiff: DiffusionSchedule, dlData, hL: Callable, oOpt, *, oScaler = None, oEma: Optional[ModelEma] = None, dropProb: float = 0.1 ) -> Tuple[float, Dict[str, float]]:

    epochLoss = 0.0
    numSamples = 0
    numBatches = len(dlData)
    gradNormSum, gradNormMax, numClipped = 0.0, 0.0, 0
    runDevice = next(oModel.parameters()).device
    oModel.train(True)

    for batchIdx, (tX, tY) in enumerate(dlData):
        tX = tX.to(runDevice, non_blocking = True) * 2 - 1
        tY = oDiff.Normalize(tY.to(runDevice, non_blocking = True))
        batchSize = tY.shape[0]
        vTime = torch.randint(oDiff.numSteps, (batchSize,), device = runDevice)
        tNoise = torch.randn(tY.shape, device = runDevice)
        tNoisy = oDiff.AddNoise(tY, vTime, tNoise)
        vCondition = (torch.rand(batchSize, device = runDevice) >= dropProb).float()

        with torch.autocast(device_type = runDevice.type, enabled = runDevice.type == 'cuda'):
            tPred = oModel(tNoisy, vTime, tX, vCondition)
        valLoss = hL(tPred.float(), oDiff.Target(tY, tNoise))
        oOpt.zero_grad()
        if oScaler is not None:
            oScaler.scale(valLoss).backward()
            oScaler.unscale_(oOpt)
            gradNorm = nn.utils.clip_grad_norm_(oModel.parameters(), 1.0).item() #<! Norm before clipping
            oScaler.step(oOpt)
            oScaler.update()
        else:
            valLoss.backward()
            gradNorm = nn.utils.clip_grad_norm_(oModel.parameters(), 1.0).item()
            oOpt.step()
        if oEma is not None:
            oEma.Update(oModel)

        gradNormSum += gradNorm
        gradNormMax = max(gradNormMax, gradNorm)
        numClipped += gradNorm > 1.0
        epochLoss += batchSize * valLoss.detach().item()
        numSamples += batchSize
        print(f'\rTrain Loss - Iteration: {(batchIdx + 1):3d} / {numBatches}, Loss: {valLoss:.6f}', end = '')

    print('\r' + ' ' * 80, end = '\r') #<! Blank the progress line so a shorter line does not leave tail characters
    dGrad = {'GradNormMean': gradNormSum / numBatches, 'GradNormMax': gradNormMax, 'ClipFraction': numClipped / numBatches}

    return epochLoss / numSamples, dGrad

@torch.inference_mode()
def EvaluateLoss( oModel: nn.Module, oDiff: DiffusionSchedule, dlData, hL: Callable, *, dropProb: float = 0.1, sampleSeed: int = 512, numBins: int = 4 ) -> Tuple[float, float, Dict[str, List[float]]]:
    # Same computation as a training step without the update; the fixed seed keeps the noise identical across epochs
    # Also the loss with a wrong (shuffled) aerial image: the gap to the true loss measures how much the network uses the condition
    # Per time bin: the training loss and the MSE of the implied clean map (the error the sampler sees before the clamp)

    runDevice = next(oModel.parameters()).device
    oGen = torch.Generator(device = runDevice).manual_seed(sampleSeed)
    oModel.eval()
    epochLoss = 0.0
    epochLossShuffled = 0.0
    numSamples = 0
    vBinLoss = torch.zeros(numBins, device = runDevice)
    vBinCleanMse = torch.zeros(numBins, device = runDevice)
    vBinCount = torch.zeros(numBins, device = runDevice)

    for tX, tY in dlData:
        tX = tX.to(runDevice, non_blocking = True) * 2 - 1
        tY = oDiff.Normalize(tY.to(runDevice, non_blocking = True))
        batchSize = tY.shape[0]
        vTime = torch.randint(oDiff.numSteps, (batchSize,), device = runDevice, generator = oGen)
        tNoise = torch.randn(tY.shape, device = runDevice, generator = oGen)
        tNoisy = oDiff.AddNoise(tY, vTime, tNoise)
        vCondition = (torch.rand(batchSize, device = runDevice, generator = oGen) >= dropProb).float()
        tTarget = oDiff.Target(tY, tNoise)

        with torch.autocast(device_type = runDevice.type, enabled = runDevice.type == 'cuda'):
            tPred = oModel(tNoisy, vTime, tX, vCondition)
            tPredShuffled = oModel(tNoisy, vTime, tX.roll(1, dims = 0), vCondition) #<! Each map paired with another image's aerial
        epochLoss += batchSize * hL(tPred.float(), tTarget).item()
        epochLossShuffled += batchSize * hL(tPredShuffled.float(), tTarget).item()
        numSamples += batchSize

        vSampleLoss = (tPred.float() - tTarget).square().mean(dim = (1, 2, 3))
        vSampleCleanMse = (oDiff.PredictClean(tNoisy, tPred.float(), vTime) - tY).square().mean(dim = (1, 2, 3))
        vBin = vTime * numBins // oDiff.numSteps
        vBinLoss.index_add_(0, vBin, vSampleLoss)
        vBinCleanMse.index_add_(0, vBin, vSampleCleanMse)
        vBinCount.index_add_(0, vBin, torch.ones_like(vSampleLoss))

    vBinCount = vBinCount.clamp(min = 1)
    dBins = {'Loss': (vBinLoss / vBinCount).tolist(), 'CleanMse': (vBinCleanMse / vBinCount).tolist()}

    return epochLoss / numSamples, epochLossShuffled / numSamples, dBins

@torch.inference_mode()
def EvaluateDiffusionModel( oModel: nn.Module, oDiff: DiffusionSchedule, dlData, hS: Callable, *, guidanceScale: float = 2.0, sampleSeed: int = 512, numGridImg: int = 8 ) -> Tuple[float, Dict[str, object], Tensor]:
    # Returns the pooled score, per image / pixel diagnostics and a grid image (rows: aerial, target, generated)

    runDevice = next(oModel.parameters()).device
    lSource, lGenerated, lTarget = [], [], []
    for batchIdx, (tX, tY) in enumerate(dlData):
        tGenerated, _, _ = SampleMaps(oModel, oDiff, tX, runDevice, guidanceScale = guidanceScale, sampleSeed = sampleSeed + batchIdx, numFrames = 0)
        lSource.append(tX.cpu())
        lGenerated.append(tGenerated)
        lTarget.append(tY.cpu())
        print(f'\rVal Map Score - Batch: {batchIdx + 1:3d} / {len(dlData)}', end = '')
    print('\r' + ' ' * 80, end = '\r')

    tSource, tGenerated, tTarget = torch.cat(lSource), torch.cat(lGenerated), torch.cat(lTarget)
    numImg = tTarget.shape[0]
    vImgScore = r2_score(tGenerated.view(numImg, -1).T, tTarget.view(numImg, -1).T, multioutput = 'raw_values') #<! Each image as an output
    vTVRatio = TotalVariation(tGenerated) / TotalVariation(tTarget).clamp(min = 1e-6) #<! < 1 blur / missing structure, > 1 speckle / blotches
    dDiag = {
        'ImageScores'     : vImgScore.tolist(),
        'ImageScoreMin'   : vImgScore.min().item(),
        'ImageScoreMedian': vImgScore.median().item(),
        'ImageScoreMax'   : vImgScore.max().item(),
        'R2'              : ImageR2Score(tGenerated, tTarget).item(),
        'SSIM'            : SSIMScore(tGenerated, tTarget).item(),
        'MSSSIM'          : MSSSIMScore(tGenerated, tTarget).item(),
        'TVAgreement'     : TVAgreementScore(tGenerated, tTarget).item(),
        'TVRatioMedian'   : vTVRatio.median().item(),
        'TVRatios'        : vTVRatio.tolist(),
        'SaturationFrac'  : ((tGenerated <= 0) | (tGenerated >= 1)).float().mean().item(),
        'GeneratedMean'   : tGenerated.mean(dim = (0, 2, 3)).tolist(),
        'GeneratedStd'    : tGenerated.std(dim = (0, 2, 3)).tolist(),
        'TargetMean'      : tTarget.mean(dim = (0, 2, 3)).tolist(),
        'TargetStd'       : tTarget.std(dim = (0, 2, 3)).tolist(),
    }
    numGridImg = min(numGridImg, numImg)
    tGrid = make_grid(torch.cat((tSource[:numGridImg], tTarget[:numGridImg], tGenerated[:numGridImg])), nrow = numGridImg, padding = 2, pad_value = 1.0)

    return hS(tGenerated, tTarget).item(), dDiag, tGrid

def TrainDiffusionModel( oModel: nn.Module, oDiff: DiffusionSchedule, dlTrain, dlVal, oOpt, numEpoch: int, hL: Callable, hS: Callable, *, oSch = None, oScaler = None, oEma: Optional[ModelEma] = None, dropProb: float = 0.1, guidanceScale: float = 2.0, valEvery: int = 5, sampleSeed: int = 512, logFolderPath: str = 'TrainLog', numGridImg: int = 8 ) -> Tuple[nn.Module, List[float], List[float], List[int], List[float], List[float]]:
    # With `oEma`, validation, scoring and the checkpoint use the averaged weights; the live weights keep training

    if valEvery < 1:
        raise ValueError('valEvery must be positive')
    os.makedirs(logFolderPath, exist_ok = True)
    oModelEval = oEma.oModel if oEma is not None else oModel
    lTrainLoss, lValLoss, lLearnRate = [], [], []
    lValEpoch, lValScore = [], []
    bestScore = -float('inf')
    totalStartTime = time.time()
    numBins = 4
    dLog = {
        'Config'         : {'NumEpochs': numEpoch, 'ValEvery': valEvery, 'GuidanceScale': guidanceScale, 'DropProb': dropProb, 'SampleSeed': sampleSeed, 'NumDiffSteps': oDiff.numSteps, 'PredictType': oDiff.predictType, 'TimeBinEdges': np.linspace(0, oDiff.numSteps, numBins + 1, dtype = int).tolist(), 'MapMean': oDiff.tMean.flatten().tolist(), 'MapStd': oDiff.dataStd, 'EmaDecay': oEma.decay if oEma is not None else None},
        'Epoch'          : {'TrainLoss': lTrainLoss, 'ValLoss': lValLoss, 'ValLossShuffled': [], 'LearnRate': lLearnRate, 'EpochTime': [], 'GradNormMean': [], 'GradNormMax': [], 'ClipFraction': [], 'ValLossBins': [], 'ValCleanMseBins': []},
        'Score'          : {'Epoch': lValEpoch, 'Score': lValScore, 'ImageScoreMin': [], 'ImageScoreMedian': [], 'ImageScoreMax': [], 'R2': [], 'SSIM': [], 'MSSSIM': [], 'TVAgreement': [], 'TVRatioMedian': [], 'TVRatios': [], 'SaturationFrac': [], 'GeneratedMean': [], 'GeneratedStd': [], 'TargetMean': [], 'TargetStd': [], 'ImageScores': []},
        'TotalTime'      : None,
    }

    for epochIdx in range(numEpoch):
        startTime = time.time()
        learnRate = oOpt.param_groups[0]['lr']
        trainLoss, dGrad = RunDiffusionEpoch(oModel, oDiff, dlTrain, hL, oOpt, oScaler = oScaler, oEma = oEma, dropProb = dropProb)
        valLoss, valLossShuffled, dBins = EvaluateLoss(oModelEval, oDiff, dlVal, hL, dropProb = dropProb, sampleSeed = sampleSeed, numBins = numBins)
        scoreEpoch = (epochIdx + 1) % valEvery == 0
        if scoreEpoch:
            valScr, dDiag, tGrid = EvaluateDiffusionModel(oModelEval, oDiff, dlVal, hS, guidanceScale = guidanceScale, sampleSeed = sampleSeed, numGridImg = numGridImg)
            lValEpoch.append(epochIdx + 1)
            lValScore.append(valScr)
            for keyName, keyVal in dDiag.items():
                dLog['Score'][keyName].append(keyVal)
        if oSch is not None:
            oSch.step()
        epochTime = time.time() - startTime

        lTrainLoss.append(trainLoss)
        lValLoss.append(valLoss)
        lLearnRate.append(learnRate)
        dLog['Epoch']['ValLossShuffled'].append(valLossShuffled)
        dLog['Epoch']['EpochTime'].append(epochTime)
        for keyName, keyVal in dGrad.items():
            dLog['Epoch'][keyName].append(keyVal)
        dLog['Epoch']['ValLossBins'].append(dBins['Loss'])
        dLog['Epoch']['ValCleanMseBins'].append(dBins['CleanMse'])
        print(f'Epoch {(epochIdx + 1):4d} / {numEpoch}', end = '')
        print(f' | Train Loss: {trainLoss:7.5f}', end = '')
        print(f' | Val Loss: {valLoss:7.5f} (Wrong Aerial: {valLossShuffled:7.5f})', end = '')
        if scoreEpoch:
            print(f' | Val Score: {valScr:6.3f} [R2: {dDiag["R2"]:6.3f}, SSIM: {dDiag["SSIM"]:.3f}, MS-SSIM: {dDiag["MSSSIM"]:.3f}, TV Ratio: {dDiag["TVRatioMedian"]:.2f}, Sat: {dDiag["SaturationFrac"]:.3f}]', end = '')
        print(f' | Epoch Time: {epochTime:5.2f}', end = '')

        if scoreEpoch:
            save_image(tGrid, os.path.join(logFolderPath, f'ValMaps_Epoch{(epochIdx + 1):04d}.png'))
        with open(os.path.join(logFolderPath, 'TrainLog.json'), 'w') as hFile:
            json.dump(dLog, hFile, indent = 2)

        if scoreEpoch and valScr > bestScore:
            bestScore = valScr
            try:
                dCheckPoint = {'Model': oModelEval.state_dict(), 'Optimizer': oOpt.state_dict(), 'MapMean': oDiff.tMean.flatten().tolist(), 'MapStd': oDiff.dataStd, 'PredictType': oDiff.predictType}
                if oEma is not None:
                    dCheckPoint['ModelRaw'] = oModel.state_dict() #<! Live weights, needed to resume training
                if oSch is not None:
                    dCheckPoint['Scheduler'] = oSch.state_dict()
                torch.save(dCheckPoint, 'BestModel.pt')
                print(' | <-- Checkpoint!', end = '')
            except OSError as oError:
                print(f' | <-- Failed: {oError}', end = '')
        print(' |')

    totalTime = time.time() - totalStartTime
    print(f'Total Training Time: {totalTime:.2f} [Sec] ({time.strftime("%H:%M:%S", time.gmtime(totalTime))})')
    dLog['TotalTime'] = totalTime
    with open(os.path.join(logFolderPath, 'TrainLog.json'), 'w') as hFile:
        json.dump(dLog, hFile, indent = 2)

    return oModel, lTrainLoss, lValLoss, lValEpoch, lValScore, lLearnRate

# %% Parameters

# Data
dataSet    = 'SatAerialToMap'
dataSetUrl = r'https://huggingface.co/datasets/Royi/DataSets/resolve/main/SatAerialToMap.zip'
imgSize = 256
trainNumSamples = None
valNumSamples = 32

# Model
baseCh = 32
lChMult = (2, 3, 6, 8) #<! Channels per level: 64 / 96 / 192 / 256; the 64 x 64 level is widened, the 16 x 16 bottleneck is unchanged
numBlocks = 2
useSeparable = False
numDiffSteps = 200
predictType = 'Clean'
conditionDropProb = 0.1
guidanceScale = 2.0

# Training
batchSize = 8
numWorkers = 4
numEpochs = 250
lossType = 'SmoothL1' #<! 'L1', 'SmoothL1', 'L2' / 'MSE'
scoreType = 'MapTVMS' #<! 'SSIM', 'MSSSIM', 'R2', 'MapTV' (R2 + SSIM + TV agreement), 'MapTVMS' (R2 + MS-SSIM + TV agreement)
valEvery = 10

# Logging
logFolder = 'TrainLog'
numGridImg = 8

# Optimizer
ηOpt = 2e-4 #<! Peak learning rate
tuβ = (0.9, 0.99)
weightDecay = 5e-5
ηMin = 5e-6 #<! Floor of the cosine decay
warmupFrac = 0.05 #<! Linear warmup, fraction of the epochs
holdFrac = 0.10 #<! Hold at the peak, fraction of the epochs (the rest is cosine decay)
emaDecay = 0.9995 #<! Per iteration; ~2000 iterations (~5 epochs) time constant. 0 disables the EMA

# %% Main Function

def Main(
    dataSet: str,
    dataSetUrl: str,
    imgSize: int,
    trainNumSamples: Optional[int],
    valNumSamples: int,
    baseCh: int,
    lChMult: Tuple[int, int, int, int],
    numBlocks: int,
    useSeparable: bool,
    numDiffSteps: int,
    predictType: Literal['Noise', 'Clean'],
    conditionDropProb: float,
    guidanceScale: float,
    batchSize: int,
    numWorkers: int,
    numEpochs: int,
    lossType: Literal['L1', 'SmoothL1', 'L2', 'MSE'],
    scoreType: Literal['SSIM', 'MSSSIM', 'R2', 'MapTV', 'MapTVMS'],
    valEvery: int,
    logFolder: str,
    numGridImg: int,
    ηOpt: float,
    tuβ: Tuple[float, float],
    weightDecay: float,
    ηMin: float,
    warmupFrac: float,
    holdFrac: float,
    emaDecay: float,
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
            # TorchVisionTrns.GaussianNoise(sigma = 0.05),
            # TorchVisionTrns.RandomErasing(p = 1.0, scale = (0.05, 0.15), ratio = (0.5, 2.0), value = 0, inplace = True),
            TorchVisionTrns.RGB(),
        ], p = [0.05, 0.05, 0.05, 0.05, 0.8]),
    ])
    oTrnsVal = TorchVisionTrns.ToDtype(torch.float32, scale = True)

    dsData = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize)
    numSamples = len(dsData)
    dsTrain = SatAerialMapDataset(datasetFolderPath, 'All', imgSize = imgSize, hTrns = oTrnsTrain, geoAug = True)
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
    print('Computing target map statistics over the training set...', end = '\r')
    tuMapMean, mapStd = ComputeMapStats(dlTrain)
    print(f'Target map mean (RGB): {tuMapMean[0]:.3f}, {tuMapMean[1]:.3f}, {tuMapMean[2]:.3f}, std: {mapStd:.3f}          ')
    print(f'Running on device: {runDevice}')

    oModel = ConditionalUNet(baseCh, numBlocks = numBlocks, lChMult = lChMult, useSeparable = useSeparable).to(runDevice)
    print(f'Model parameters: {sum(p.numel() for p in oModel.parameters()) / 1e6:.2f} [M]')
    oDiff = DiffusionSchedule(numDiffSteps, runDevice, vMean = tuMapMean, dataStd = mapStd, predictType = predictType)
    hL = Pix2PixLoss(lossType = lossType).to(runDevice)
    hS = Pix2PixScore(scoreType = scoreType).to(runDevice)
    oOpt = torch.optim.AdamW(oModel.parameters(), lr = ηOpt, betas = tuβ, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.LambdaLR(oOpt, WarmupHoldCosine(numEpochs, warmupFrac, holdFrac, ηMin / ηOpt))
    oScaler = torch.amp.GradScaler('cuda', enabled = runDevice.type == 'cuda')
    oEma = ModelEma(oModel, emaDecay) if emaDecay > 0 else None

    TrainDiffusionModel(oModel, oDiff, dlTrain, dlVal, oOpt, numEpochs, hL, hS, oSch = oSch, oScaler = oScaler, oEma = oEma, dropProb = conditionDropProb, guidanceScale = guidanceScale, valEvery = valEvery, sampleSeed = seedNum, logFolderPath = logFolder, numGridImg = numGridImg)

# %% Main

if __name__ == '__main__':
    Main(dataSet, dataSetUrl, imgSize, trainNumSamples, valNumSamples,
         baseCh, lChMult, numBlocks, useSeparable, numDiffSteps, predictType, conditionDropProb, guidanceScale,
         batchSize, numWorkers, numEpochs, lossType, scoreType, valEvery, logFolder, numGridImg,
         ηOpt, tuβ, weightDecay, ηMin, warmupFrac, holdFrac, emaDecay)