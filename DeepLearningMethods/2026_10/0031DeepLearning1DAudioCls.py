# %% [markdown]
#
# # Deep Learning Methods
#
# ## Deep Learning - 1D Convolution Net for Audio Classification
#
# Training script for 0031DeepLearning1DAudioCls.ipynb.
# Run from the course directory. BestModel.pt contains the best model
# and the complete training histories for the notebook's loading cell.

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Deep Learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchmetrics.classification import MulticlassAccuracy

# Miscellaneous
from concurrent.futures import ThreadPoolExecutor
import os
import random

# Typing
from typing import Callable, Literal, Optional, Self, Sequence, Tuple, Union
from numpy.typing import NDArray
from torch import Tensor

# %% Configuration

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

torch.backends.cudnn.benchmark = True

# %% Constants

PROJECT_NAME       = 'FixelCourses'
DATA_FOLDER_NAME   = 'DataSets'
BASE_FOLDER_PATH   = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH   = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)

# %% Course Packages

from DataManipulation import DownloadUrl
from DeepLearningPyTorch import TrainModel

# %% Auxiliary Functions

class AudioMNISTDataset(Dataset):
    def __init__( self, dfData: pd.DataFrame, targetCol: Literal['Digit', 'Speaker', 'Accent', 'Gender', 'NativeSpeaker'], hMap: Callable[[Union[int, str]], int], *, hTransform: Optional[Callable] = None, padSize: int = 0, seedNum: Optional[int] = None ) -> None:

        signalStartIdx = dfData.columns.tolist().index('0')

        # Match the notebook's Int16 conversion without copying a full DataFrame per worker.
        self._mData      = dfData.iloc[:, signalStartIdx:].to_numpy().astype(np.int16)
        self._vN         = dfData['NumSamples'].to_numpy(copy = True)
        self._vY         = dfData[targetCol].to_numpy(copy = True)
        self._dsSpeaker  = dfData['Speaker'].copy()
        self._hTransform = hTransform
        self._hMap       = hMap
        self._winSize    = dfData.shape[1] + padSize
        self._oRng       = random.Random(seedNum)

    def __len__( self: Self ) -> int:

        return len(self._vY)

    def __getitem__( self: Self, idx: int ) -> Tuple[Union[NDArray, Tensor], int]:

        vA = self._mData[idx, :self._vN[idx]]
        vX = np.zeros(self._winSize, dtype = np.float32)
        maxShift = self._winSize - len(vA)

        rndShift = self._oRng.randint(0, maxShift)
        vX[rndShift:(rndShift + len(vA))] = vA.astype(np.float32)
        vX = np.expand_dims(vX, axis = 0)

        valY = self._hMap(self._vY[idx])

        if self._hTransform is not None:
            vX = self._hTransform(vX)

        return vX, valY

    def GenTrainValSplit( self: Self, valFraction: float ) -> Tuple[Sequence[int], Sequence[int]]:

        lSpeakerIdx = self._dsSpeaker.unique().tolist()
        numSpeakers = len(lSpeakerIdx)

        numValSpeakers   = int(np.floor(valFraction * numSpeakers))
        numTrainSpeakers = numSpeakers - numValSpeakers

        lTrainSpkIdx = self._oRng.sample(lSpeakerIdx, k = numTrainSpeakers)
        lValSpkIdx   = [spk for spk in lSpeakerIdx if spk not in lTrainSpkIdx]

        lTrainIdx = self._dsSpeaker[self._dsSpeaker.isin(lTrainSpkIdx)].index.tolist()
        lValIdx   = self._dsSpeaker[self._dsSpeaker.isin(lValSpkIdx)].index.tolist()

        return lTrainIdx, lValIdx

    def GetWinSize( self: Self ) -> int:

        return self._winSize


def InitAudioWorker( workerIdx: int ) -> None:

    oWorkerInfo = torch.utils.data.get_worker_info()
    if oWorkerInfo is None:
        raise RuntimeError('Audio worker initialization requires a DataLoader worker.')

    dsData = oWorkerInfo.dataset.dataset
    dsData._oRng.seed(oWorkerInfo.seed)


def DownloadUrlFile( fileUrl: str, destFolderPath: str, fileName: str ) -> str:

    os.makedirs(destFolderPath, exist_ok = True)

    filePath = os.path.join(destFolderPath, fileName)
    filePath = DownloadUrl(fileUrl, filePath)

    return filePath


def DecimateDataThread( ii: int, mDDec: NDArray, vNDec: NDArray, mD: NDArray, vN: NDArray, decFactor: int ) -> None:

    vA    = mD[ii, :vN[ii]]
    vADec = sp.signal.decimate(vA, decFactor)

    numSamplesDec = len(vADec)
    mDDec[ii, :numSamplesDec] = vADec
    vNDec[ii]                 = numSamplesDec


def DecimateDataParallel( mD: NDArray, vN: NDArray, decFactor: int, winSize: int, numWorkers: int ) -> Tuple[NDArray, NDArray]:

    numSignals = mD.shape[0]
    mDDec = np.zeros((numSignals, winSize))
    vNDec = np.zeros(numSignals, dtype = np.int32)

    hDecimateDataThread = lambda ii: DecimateDataThread(ii, mDDec, vNDec, mD, vN, decFactor)

    with ThreadPoolExecutor(max_workers = max(1, numWorkers)) as hExe:
        for _ in hExe.map(hDecimateDataThread, range(numSignals)):
            pass #<! Consume results so worker errors propagate.

    return mDDec, vNDec

# %% Parameters

# Data
dataSetName     = 'AudioMNIST'
datasetFileUrl  = r'https://huggingface.co/datasets/Royi/AudioMNIST/resolve/main/AudioMNIST.parquet'
datasetFileName = 'AudioMNIST.parquet'
targetCol       = 'Digit'
numCls          = 10

decFactor = 8
winSize   = 6_000

# Dataset
padSize     = 400
valSetRatio = 0.2

# Model
dropP = 0.1

# Training
batchSize    = 512
numWorkers   = 4
nEpochs      = 20
learningRate = 5e-4
weightDecay  = 5e-5
tuBeta       = (0.9, 0.99)

# %% Main Function

def Main() -> None:

    datasetFolderPath = os.path.join(DATA_FOLDER_PATH, dataSetName)
    datasetFilePath = DownloadUrlFile(datasetFileUrl, datasetFolderPath, datasetFileName)

    dfData = pd.read_parquet(datasetFilePath)
    signalStartIdx = dfData.columns.tolist().index('0')
    mData = dfData.iloc[:, signalStartIdx:].to_numpy()
    vN = dfData['NumSamples'].to_numpy()
    mDataDec, vNDec = DecimateDataParallel(mData, vN, decFactor, winSize, numWorkers)

    dfAudioDec = pd.DataFrame(mDataDec, columns = [str(ii) for ii in range(mDataDec.shape[1])])
    dfData = pd.concat((dfData.iloc[:, :signalStartIdx].copy(), dfAudioDec), axis = 1)
    dfData['NumSamples'] = vNDec
    del mData, vN, mDataDec, vNDec, dfAudioDec

    dfData['Accent'] = dfData['Accent'].replace({'german': 'German', 'Egyptian_American?': 'Egyptian/American'})
    dfData['Age'] = dfData['Age'].replace({1234: 34})
    dfData['Continent'] = dfData['Continent'].replace({'South-America': 'America'})

    dsData = AudioMNISTDataset(dfData, targetCol, hMap = int, padSize = padSize, seedNum = seedNum)
    lTrainIdx, lValIdx = dsData.GenTrainValSplit(valSetRatio)
    dsTrain = torch.utils.data.Subset(dsData, lTrainIdx)
    dsVal   = torch.utils.data.Subset(dsData, lValIdx)
    del dfData

    persistentWorkers = numWorkers > 0
    pinMemory = torch.cuda.is_available()
    dlTrain = torch.utils.data.DataLoader(dsTrain, batch_size = batchSize, shuffle = True, num_workers = numWorkers, pin_memory = pinMemory, drop_last = True, persistent_workers = persistentWorkers, worker_init_fn = InitAudioWorker)
    dlVal   = torch.utils.data.DataLoader(dsVal, batch_size = 2 * batchSize, shuffle = False, num_workers = numWorkers, pin_memory = pinMemory, drop_last = False, persistent_workers = persistentWorkers, worker_init_fn = InitAudioWorker)

    oModel = nn.Sequential(
        nn.Identity(),

        nn.Conv1d(in_channels = 1,   out_channels = 16,  kernel_size = 21), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 16,  out_channels = 32,  kernel_size = 19), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 32,  out_channels = 48,  kernel_size = 17), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 48,  out_channels = 64,  kernel_size = 15), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 64,  out_channels = 96,  kernel_size = 13), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 96,  out_channels = 128, kernel_size = 11), nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 128, out_channels = 160, kernel_size = 9) , nn.MaxPool1d(kernel_size = 2), nn.ReLU(),
        nn.Conv1d(in_channels = 160, out_channels = 192, kernel_size = 7) , nn.MaxPool1d(kernel_size = 2), nn.ReLU(),

        nn.AdaptiveAvgPool1d(output_size = 1), nn.ReLU(),
        nn.Flatten          (),
        nn.Linear           (in_features = 192, out_features = 128), nn.ReLU(),
        nn.Dropout          (p = dropP),
        nn.Linear           (in_features = 128, out_features = 64),  nn.ReLU(),
        nn.Dropout          (p = dropP),
        nn.Linear           (in_features = 64, out_features = numCls),
    )

    runDevice = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f'Running on device: {runDevice}')
    print(f'DataLoader workers: {numWorkers}')

    oModel = oModel.to(runDevice)
    hL = nn.CrossEntropyLoss()
    hS = MulticlassAccuracy(num_classes = numCls, average = 'micro').to(runDevice)

    oOpt = torch.optim.AdamW(oModel.parameters(), lr = learningRate, betas = tuBeta, weight_decay = weightDecay)
    oSch = torch.optim.lr_scheduler.OneCycleLR(oOpt, max_lr = learningRate, total_steps = nEpochs)

    checkpointPath = 'BestModel.pt'
    checkpointTime = os.stat(checkpointPath).st_mtime_ns if os.path.isfile(checkpointPath) else None

    _, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate = TrainModel(oModel, dlTrain, dlVal, oOpt, nEpochs, hL, hS, oSch = oSch)

    if not os.path.isfile(checkpointPath) or os.stat(checkpointPath).st_mtime_ns == checkpointTime:
        raise RuntimeError('Training did not save a new BestModel.pt checkpoint.')

    dModel = torch.load(checkpointPath, map_location = 'cpu', weights_only = True)
    dModel.update({
        'TrainLoss': lTrainLoss,
        'TrainScore': lTrainScore,
        'ValLoss': lValLoss,
        'ValScore': lValScore,
        'LearnRate': lLearnRate,
    })
    torch.save(dModel, checkpointPath)
    print(f'Best model and training histories saved to: {os.path.abspath(checkpointPath)}')

# %% Main

if __name__ == '__main__':
    Main()
