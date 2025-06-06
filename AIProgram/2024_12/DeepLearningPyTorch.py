
# Python STD
from enum import auto, Enum, unique
import math

# Data
import numpy as np
# import pandas as pd
import scipy as sp

# Machine Learning

# Deep Learning
import torch
import torch.nn            as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

# Image Processing / Computer Vision

# Optimization

# Python STD
import os

# Auxiliary

# Visualization
import matplotlib.pyplot as plt

# Miscellaneous
import time

# Course Packages
from DeepLearningBlocks import NNMode


# Typing
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union

# Auxiliary Classes

class TBLogger():
    def __init__( self, logDir: Optional[str] = None ) -> None:

        self.oTBWriter  = SummaryWriter(log_dir = logDir)
        self.iiEpcoh    = 0
        self.iiItr      = 0
        
        pass

    def close( self ) -> None:

        self.oTBWriter.close()

class TestDataSet( torchvision.datasets.VisionDataset ):
    def __init__(self, root: str = None, transforms: Callable[..., Any] | None = None, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        lF = os.listdir(root)
        lFiles = [fileName for fileName in lF if (os.path.isfile(os.path.join(root, fileName)) and (os.path.splitext(os.path.join(root, fileName))[1] in IMG_EXTENSIONS))]

        self.lFiles = lFiles
        self.loader = pil_loader
    
    def __len__(self) -> int:
        
        return len(self.lFiles)
    
    def __getitem__(self, index: int) -> Any:
        
        imgSample =  self.loader(os.path.join(self.root, self.lFiles[index]))
        if self.transform is not None:
            imgSample = self.transform(imgSample)
        
        return imgSample

class ObjectLocalizationDataset( Dataset ):
    def __init__( self, tX: np.ndarray, vY: np.ndarray, mB: np.ndarray, singleY: bool = True ) -> None:

        if (tX.shape[0] != vY.shape[0]):
            raise ValueError(f'The number of samples in `tX` and `vY` does not match!')
        if (tX.shape[0] != mB.shape[0]):
            raise ValueError(f'The number of samples in `tX` and `mB` does not match!')
        
        self.tX         = tX #<! (numSamples, H, W, C)
        self.vY         = vY #<! (numSamples, )
        self.mB         = mB #<! (numSamples, 4)
        self.singleY    = singleY #<! Return label and box, or a single vector
        self.numSamples = tX.shape[0]

    def __len__( self: Self ) -> int:
        
        return self.numSamples

    def __getitem__( self: Self, idx: int ) -> Union[Tuple[np.ndarray, int, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        
        tXi   = self.tX[idx] #<! Image
        valYi = self.vY[idx] #<! Label
        vBi   = self.mB[idx] #<! Bounding Box

        tXi   = tXi.astype(np.float32)
        vBi   = vBi.astype(np.float32)

        if self.singleY:
            valYi = valYi.astype(np.float32)
            return tXi, np.r_[valYi, vBi]
        else:
            return tXi, valYi, vBi

class ObjectDetectionDataset( Dataset ):
    def __init__( self, tX: np.ndarray, lY: List[np.ndarray], lB: List[np.ndarray], hDataTrans: Optional[Callable] = None ) -> None:

        if (tX.shape[0] != len(lY)):
            raise ValueError(f'The number of samples in `tX` and `lY` does not match!')
        if (tX.shape[0] != len(lB)):
            raise ValueError(f'The number of samples in `tX` and `lB` does not match!')
        
        self.tX = tX
        self.lY = lY
        self.lB = lB
        self.numSamples = tX.shape[0]
        self.hDataTrans = hDataTrans

    def __len__( self: Self ) -> int:
        
        return self.numSamples

    def __getitem__( self: Self, idx: int ) -> Union[Tuple[np.ndarray, int, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        
        tXi = self.tX[idx] #<! Image
        vYi = self.lY[idx] #<! Labels
        mBi = self.lB[idx] #<! Bounding Boxes

        tXi = tXi.astype(np.float32)
        vYi = vYi.astype(np.float32)
        mBi = mBi.astype(np.float32)

        mYi = np.c_[vYi, mBi]

        if self.hDataTrans is not None:
            tXi, mYi = self.hDataTrans(tXi, mYi)
        
        return tXi, mYi

# Auxiliary Functions

def ResetModelWeights( oModel: nn.Module ) -> None:
    # https://discuss.pytorch.org/t/19180

    for layer in oModel.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif hasattr(layer, 'children'):
            for child in layer.children():
                ResetModelWeights(child)

# Initialization Function

def InitWeights( oLayer: nn.Module ) -> None:
    # Use: oModel.apply(InitWeights) #<! Applies the function on all layers
    if isinstance(oLayer, nn.Linear):
        nn.init.kaiming_normal_(oLayer.weight.data)

def InitWeightsKaiNorm( oLayer: nn.Module, tuLyrClas: Tuple = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Bilinear, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d) ) -> None:
    # Use: oModel.apply(InitWeights) #<! Applies the function on all layers
    if isinstance(oLayer, tuLyrClas):
        nn.init.kaiming_normal_(oLayer.weight.data)


def GenDataLoaders( dsTrain: Dataset, dsVal: Dataset, batchSize: int, *, numWorkers: int = 0, CollateFn: Callable = default_collate, dropLast: bool = True, PersWork: bool = False ) -> Tuple[DataLoader, DataLoader]:

    if numWorkers == 0: 
        PersWork = False

    dlTrain = torch.utils.data.DataLoader(dsTrain, shuffle = True, batch_size = 1 * batchSize, num_workers = numWorkers, collate_fn = CollateFn, drop_last = dropLast, persistent_workers = PersWork)
    dlVal   = torch.utils.data.DataLoader(dsVal, shuffle = False, batch_size = 2 * batchSize, num_workers = numWorkers, persistent_workers = PersWork)

    return dlTrain, dlVal

def GenResNetModel( trainedModel: bool, numCls: int, resNetDepth: Literal[18, 34, 50] = 18 ) -> nn.Module:
    # Read on the API change at: How to Train State of the Art Models Using TorchVision’s Latest Primitives
    # https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives

    match resNetDepth:
        case 18:
            modelFun = torchvision.models.resnet18
            modelWeights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        case 34:
            modelFun = torchvision.models.resnet34
            modelWeights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        case 50:
            modelFun = torchvision.models.resnet50
            modelWeights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        case 101:
            modelFun = torchvision.models.resnet101
            modelWeights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
        case 152:
            modelFun = torchvision.models.resnet152
            modelWeights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
        case _:
            raise ValueError(f'The `resNetDepth`: {resNetDepth} is invalid!')

    if trainedModel:
        oModel        = modelFun(weights = modelWeights)
        numFeaturesIn = oModel.fc.in_features
        # Assuming numCls << 1000
        oModel.fc     = nn.Sequential(
            nn.Linear(numFeaturesIn, 128), nn.ReLU(),
            nn.Linear(128, numCls),
        )
    else:
        oModel = modelFun(weights = None, num_classes = numCls)

    return oModel

def RunEpoch( oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, opMode: NNMode = NNMode.TRAIN ) -> Tuple[float, float]:
    """
    Runs a single Epoch (Train / Test) of a model.  
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlData      - PyTorch `Dataloader` object.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oOpt        - PyTorch `Optimizer` object.
        opMode      - An `NNMode` to set the mode of operation.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer is required for training mode.
    """
    
    epochLoss   = 0.0
    epochScore  = 0.0
    numSamples  = 0
    numBatches = len(dlData)

    runDevice = next(oModel.parameters()).device #<! CPU \ GPU

    if opMode == NNMode.TRAIN:
        oModel.train(True) #<! Equivalent of `oModel.train()`
        trainMode = True
    elif opMode == NNMode.INFERENCE:
        oModel.eval() #<! Equivalent of `oModel.train(False)`
        trainMode = False
    else:
        raise ValueError(f'The `opMode` value {opMode} is not supported!')
    
    for ii, (mX, vY) in enumerate(dlData):
        # Move Data to Model's device
        mX = mX.to(runDevice) #<! Lazy
        vY = vY.to(runDevice) #<! Lazy

        batchSize = mX.shape[0]
        
        if opMode == NNMode.TRAIN:
            # Forward
            mZ      = oModel(mX) #<! Model output
            valLoss = hL(mZ, vY) #<! Loss
            
            # Backward
            oOpt.zero_grad()    #<! Set gradients to zeros
            valLoss.backward()  #<! Backward
            oOpt.step()         #<! Update parameters
            oModel.eval()       #<! Inference mode for layers
        else: #<! Value of `opMode` was already validated
            with torch.inference_mode(): #<! The `torch.inference_mode()` scope is more optimized than `torch.no_grad()` 
                # No computational graph
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss

        with torch.inference_mode():
            # Score
            oModel.eval() #<! Ensure Evaluation Mode (Dropout / Normalization layers)
            valScore = hS(mZ, vY)
            # Normalize so each sample has the same weight
            epochLoss  += batchSize * valLoss.item()
            epochScore += batchSize * valScore.item()
            numSamples += batchSize
            oModel.train(trainMode) #<! Restore original mode

        print(f'\r{"Train" if trainMode else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}, loss: {valLoss:.6f}', end = '')
    
    print('', end = '\r')
            
    return epochLoss / numSamples, epochScore / numSamples

# Training Epoch
def RunEpochSch( oModel: nn.Module, dlData: DataLoader, hL: Callable, hS: Callable, oOpt: Optional[Optimizer] = None, oSch: Optional[LRScheduler] = None, opMode: NNMode = NNMode.TRAIN, oTBLogger: Optional[TBLogger] = None ) -> Tuple[float, float]:
    """
    Runs a single Epoch (Train / Test) of a model.  
    Supports per iteration (Batch) scheduling. 
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlData      - PyTorch `Dataloader` object.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oOpt        - PyTorch `Optimizer` object.
        oSch        - PyTorch `Scheduler` (`LRScheduler`) object.
        opMode      - An `NNMode` to set the mode of operation.
        oTBLogger   - An `TBLogger` object.
    Output:
        valLoss     - Scalar of the loss.
        valScore    - Scalar of the score.
        learnRate   - Scalar of the average learning rate over the epoch.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer / scheduler are required for training mode.
    """
    
    epochLoss   = 0.0
    epochScore  = 0.0
    numSamples  = 0
    epochLr     = 0.0
    numBatches = len(dlData)
    lLearnRate = []

    runDevice = next(oModel.parameters()).device #<! CPU \ GPU

    if opMode == NNMode.TRAIN:
        oModel.train(True) #<! Equivalent of `oModel.train()`
        trainMode = True
    elif opMode == NNMode.INFERENCE:
        oModel.eval() #<! Equivalent of `oModel.train(False)`
        trainMode = False
    else:
        raise ValueError(f'The `opMode` value {opMode} is not supported!')
    
    for ii, (mX, vY) in enumerate(dlData):
        # Move Data to Model's device
        mX = mX.to(runDevice) #<! Lazy
        vY = vY.to(runDevice) #<! Lazy

        batchSize = mX.shape[0]
        
        if opMode == NNMode.TRAIN:
            # Forward
            mZ      = oModel(mX) #<! Model output
            valLoss = hL(mZ, vY) #<! Loss
            
            # Backward
            oOpt.zero_grad()    #<! Set gradients to zeros
            valLoss.backward()  #<! Backward
            oOpt.step()         #<! Update parameters

            learnRate = oSch.get_last_lr()[0]
            oSch.step()         #<! Update learning rate
            oModel.eval()       #<! Inference mode for layers
        else: #<! Value of `opMode` was already validated
            with torch.no_grad():
                # No computational graph
                mZ      = oModel(mX) #<! Model output
                valLoss = hL(mZ, vY) #<! Loss
                
                learnRate = 0.0

        with torch.no_grad():
            # Score
            valScore = hS(mZ, vY)
            # Normalize so each sample has the same weight
            epochLoss  += batchSize * valLoss.item()
            epochScore += batchSize * valScore.item()
            epochLr    += batchSize * learnRate
            numSamples += batchSize
            lLearnRate.append(learnRate)

            oModel.train(trainMode) #<! Restore original mode

            if (oTBLogger is not None) and trainMode:
                # Logging at Iteration level for training
                oTBLogger.iiItr += 1
                oTBLogger.oTBWriter.add_scalar('Train Loss', valLoss.item(), oTBLogger.iiItr)
                oTBLogger.oTBWriter.add_scalar('Train Score', valScore.item(), oTBLogger.iiItr)
                oTBLogger.oTBWriter.add_scalar('Learning Rate', learnRate, oTBLogger.iiItr)

        print(f'\r{"Train" if trainMode else "Val"} - Iteration: {(ii + 1):3d} / {numBatches}, loss: {valLoss:.6f}', end = '')
    
    print('', end = '\r')
            
    return epochLoss / numSamples, epochScore / numSamples, epochLr / numSamples, lLearnRate

# Training Model Loop Function

def TrainModel( oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, numEpoch: int, hL: Callable, hS: Callable, *, oSch: Optional[LRScheduler] = None, oTBWriter: Optional[SummaryWriter] = None) -> Tuple[nn.Module, List, List, List, List]:
    """
    Trains a model given test and validation data loaders.  
    Input:
        oModel      - PyTorch `nn.Module` object.
        dlTrain     - PyTorch `Dataloader` object (Training).
        dlVal       - PyTorch `Dataloader` object (Validation).
        oOpt        - PyTorch `Optimizer` object.
        numEpoch    - Number of epochs to run.
        hL          - Callable for the Loss function.
        hS          - Callable for the Score function.
        oSch        - PyTorch `Scheduler` (`LRScheduler`) object.
        oTBWriter   - PyTorch `SummaryWriter` object (TensorBoard).
    Output:
        lTrainLoss     - Scalar of the loss.
        lTrainScore    - Scalar of the score.
        lValLoss    - Scalar of the score.
        lValScore    - Scalar of the score.
        lLearnRate    - Scalar of the score.
    Remarks:
      - The `oDataSet` object returns a Tuple of (mX, vY) per batch.
      - The `hL` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a Tuple of `valLoss` (Scalar of the loss) and `mDz` (Gradient by the loss).
      - The `hS` function should accept the `vY` (Reference target) and `mZ` (Output of the NN).  
        It should return a scalar `valScore` of the score.
      - The optimizer is required for training mode.
    """

    lTrainLoss  = []
    lTrainScore = []
    lValLoss    = []
    lValScore   = []
    lLearnRate  = []

    # Support R2
    bestScore = -1e9 #<! Assuming higher is better

    learnRate = oOpt.param_groups[0]['lr']

    for ii in range(numEpoch):
        startTime           = time.time()
        trainLoss, trainScr = RunEpoch(oModel, dlTrain, hL, hS, oOpt, opMode = NNMode.TRAIN) #<! Train
        valLoss,   valScr   = RunEpoch(oModel, dlVal, hL, hS, oOpt, opMode = NNMode.INFERENCE) #<! Score Validation
        if oSch is not None:
            # Adjusting the scheduler on Epoch level
            learnRate = oSch.get_last_lr()[0]
            oSch.step()
        epochTime           = time.time() - startTime

        # Aggregate Results
        lTrainLoss.append(trainLoss)
        lTrainScore.append(trainScr)
        lValLoss.append(valLoss)
        lValScore.append(valScr)
        lLearnRate.append(learnRate)

        if oTBWriter is not None:
            oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
            oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
            oTBWriter.add_scalar('Learning Rate', learnRate, ii)
        
        # Display (Babysitting)
        print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}', end = '')
        print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
        print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
        print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
        print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
        print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

        # Save best model ("Early Stopping")
        if valScr > bestScore:
            bestScore = valScr
            try:
                dCheckPoint = {'Model': oModel.state_dict(), 'Optimizer': oOpt.state_dict()}
                if oSch is not None:
                    dCheckPoint['Scheduler'] = oSch.state_dict()
                torch.save(dCheckPoint, 'BestModel.pt')
                print(' | <-- Checkpoint!', end = '')
            except:
                print(' | <-- Failed!', end = '')
        print(' |')
    
    # Load best model ("Early Stopping")
    # dCheckPoint = torch.load('BestModel.pt')
    # oModel.load_state_dict(dCheckPoint['Model'])

    return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate


def TrainModelSch( oModel: nn.Module, dlTrain: DataLoader, dlVal: DataLoader, oOpt: Optimizer, oSch: LRScheduler, numEpoch: int, hL: Callable, hS: Callable, oTBLogger: Optional[TBLogger] = None ) -> Tuple[nn.Module, List, List, List, List]:

    lTrainLoss  = []
    lTrainScore = []
    lValLoss    = []
    lValScore   = []
    lLearnRate  = []

    # Support R2
    bestScore = -1e9 #<! Assuming higher is better

    for ii in range(numEpoch):
        startTime                               = time.time()
        trainLoss, trainScr, trainLr, lLRate    = RunEpochSch(oModel, dlTrain, hL, hS, oOpt, oSch, opMode = NNMode.TRAIN, oTBLogger = oTBLogger) #<! Train
        valLoss,   valScr, *_                   = RunEpochSch(oModel, dlVal, hL, hS, opMode = NNMode.INFERENCE)    #<! Score Validation
        epochTime                               = time.time() - startTime

        # Aggregate Results
        lTrainLoss.append(trainLoss)
        lTrainScore.append(trainScr)
        lValLoss.append(valLoss)
        lValScore.append(valScr)
        lLearnRate.extend(lLRate)

        if oTBLogger is not None:
            oTBLogger.iiEpcoh += 1
            oTBLogger.oTBWriter.add_scalars('Loss (Epoch)', {'Train': trainLoss, 'Validation': valLoss}, ii)
            oTBLogger.oTBWriter.add_scalars('Score (Epoch)', {'Train': trainScr, 'Validation': valScr}, ii)
            oTBLogger.oTBWriter.add_scalar('Learning Rate (Epoch)', trainLr, ii)
            oTBLogger.oTBWriter.flush()
        
        # Display (Babysitting)
        print('Epoch '              f'{(ii + 1):4d} / ' f'{numEpoch}', end = '')
        print(' | Train Loss: '     f'{trainLoss          :6.3f}', end = '')
        print(' | Val Loss: '       f'{valLoss            :6.3f}', end = '')
        print(' | Train Score: '    f'{trainScr           :6.3f}', end = '')
        print(' | Val Score: '      f'{valScr             :6.3f}', end = '')
        print(' | Epoch Time: '     f'{epochTime          :5.2f}', end = '')

        # Save best model ("Early Stopping")
        if valScr > bestScore:
            bestScore = valScr
            print(' | <-- Checkpoint!', end = '')
            try:
                dCheckpoint = {'Model' : oModel.state_dict(), 'Optimizer' : oOpt.state_dict(), 'Scheduler': oSch.state_dict()}
                torch.save(dCheckpoint, 'BestModel.pt')
            except:
                print(' | <-- Failed!', end = '')
        print(' |')
    
    # Load best model ("Early Stopping")
    dCheckpoint = torch.load('BestModel.pt')
    oModel.load_state_dict(dCheckpoint['Model'])

    return oModel, lTrainLoss, lTrainScore, lValLoss, lValScore, lLearnRate

# Auxiliary Blocks

# Residual Class
# Residual Block:
# - https://scribe.rip/471810e894ed.
# - https://stackoverflow.com/questions/57229054.
# - https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
# For nn. vs. F. see:
# - https://discuss.pytorch.org/t/31857
# - https://stackoverflow.com/questions/53419474

# Simple Residual Block
class ResidualBlock( nn.Module ):
    def __init__( self, numChnl: int ) -> None:
        super(ResidualBlock, self).__init__()
        
        self.oConv2D1       = nn.Conv2d(numChnl, numChnl, kernel_size = 3, padding = 1, bias = False)
        self.oBatchNorm1    = nn.BatchNorm2d(numChnl)
        self.oReLU1         = nn.ReLU(inplace = True)
        self.oConv2D2       = nn.Conv2d(numChnl, numChnl, kernel_size = 3, padding = 1, bias = False)
        self.oBatchNorm2    = nn.BatchNorm2d(numChnl)
        self.oReLU2         = nn.ReLU(inplace = True) #<! No need for it, for better visualization
            
    def forward( self: Self, tX: torch.Tensor ) -> torch.Tensor:
        
        tY = self.oReLU1(self.oBatchNorm1(self.oConv2D1(tX)))
        tY = self.oBatchNorm2(self.oConv2D2(tY))
        tY += tX
        tY = self.oReLU2(tY)
		
        return tY
    

class YoloGrid( nn.Module ):
    def __init__( self, gridSize: int ) -> None:
        super(YoloGrid, self).__init__()
        
        self.gridSize = gridSize
            
    def forward( self: Self, tX: torch.Tensor, tB: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts YOLO bounding box into a target grid.  
        Input:
            tX          - Image (C x H x W).
            tB          - Bounding Box (D x 5).
        Output:
            tX          - Image (C x H x W).
            tY          - Target (6 x S x S).
        Remarks:
          - The grid size is `S x S`.
          - Input target: tB[ii, :] -> [cls, x, y, W, H].
          - Output target: tY[:, ii, jj] -> [prob, x, y, W, H, cls].
        """
        
        D        = tB.shape[0] #<! Number of Boxes
        S        = self.gridSize
        mProb    = torch.zeros(1, S, S)
        mLabel   = torch.zeros(1, S, S)
        mBBoxOut = torch.zeros(4, S, S)
        
        if D > 0:
            vCls            = tB[:, :1].T
            mXYWH           = tB[:, 1:] * S
            vX, vY, vW ,vH  = mXYWH.T
            vCx             = vX.floor().long() #<! Cell x index
            vCy             = vY.floor().long() #<! Cell y index
            vX             -= vCx               #<! Cell x
            vY             -= vCy               #<! Cell y
            
            mProb[0, vCy, vCx]      = 1.0
            mLabel[0, vCy, vCx]     = vCls
            mBBoxOut[:, vCy, vCx]   = torch.stack([vX, vY, vW, vH])
        
        tY = torch.cat([mProb, mBBoxOut, mLabel])
		
        return tX, tY

class YoloBox( nn.Module ):
    def __init__( self, gridSize: int ) -> None:
        super(YoloBox, self).__init__()

        self.gridSize = gridSize
            
    def forward( self: Self, tX: torch.Tensor, tY: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts YOLO target grid into a bounding box.  
        Input:
            tX          - Image (C x H x W).
            tY          - Target (6 x S x S).
        Output:
            tX          - Image (C x H x W).
            tB          - Bounding Box (D x 5).
        Remarks:
          - The grid size is `S x S`.
          - Input target: tB[ii, :] -> [cls, x, y, W, H].
          - Output target: tY[:, ii, jj] -> [prob, x, y, W, H, cls].
        """

        if ((tY.shape[1] != self.gridSize) or (tY.shape[2] != self.gridSize)):
            raise ValueError('The dimensions of the grid size does not match the dimensions of `tY`')
        
        S        = self.gridSize
        vG       = torch.arange(S)
        mXX, mYY = torch.meshgrid(vG, vG, indexing = 'xy')

        mProb  = tY[:, [0], :, :]
        mXYWH  = tY[:, 1:5, :, :].clone()
        mLabel = tY[:, [5], :, :]

        mXYWH[:, 0, :, :] += mXX[None, :, :]
        mXYWH[:, 1, :, :] += mYY[None, :, :]
        
        tB = torch.cat([mProb, mLabel, mXYWH], dim = 1)   #<! mBBox.shape = (N, 6,  5, 5)
        tB = tB.permute(0, 2, 3, 1).reshape(-1, S * S, 6) #<! mBBox.shape = (N, 25, 6)
		
        return tX, tB

class NetToTgt( nn.Module ):
    def __init__( self, gridSize: int, numCls: int ) -> None:
        super(NetToTgt, self).__init__()

        self.gridSize = gridSize
        self.numCls   = numCls
            
    def forward( self: Self, tX: torch.Tensor, tY: torch.Tensor ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts net (Model) output to YOLO Grid format.  
        Input:
            tX          - Image (N x C x H x W).
            tY          - Target (N x (5 + numCls) x S x S).
        Output:
            tX          - Image (C x H x W).
            tB          - Bounding Box (D x 5).
        Remarks:
          - The grid size is `S x S`.
          - Input target: tB[ii, :] -> [cls, x, y, W, H].
          - Output target: tY[:, ii, jj] -> [prob, x, y, W, H, cls].
        """

        # TODO: Complete!

        if ((tY.shape[1] != self.gridSize) or (tY.shape[2] != self.gridSize)):
            raise ValueError('The dimensions of the grid size does not match the dimensions of `tY`')
        
        S        = self.gridSize
        vG       = torch.arange(S)
        mXX, mYY = torch.meshgrid(vG, vG, indexing = 'xy')

        mProb  = tY[:, [0], :, :]
        mXYWH  = tY[:, 1:5, :, :].clone()
        mLabel = tY[:, [5], :, :]

        mXYWH[:, 0, :, :] += mXX[None, :, :]
        mXYWH[:, 1, :, :] += mYY[None, :, :]
        
        tB = torch.cat([mProb, mLabel, mXYWH], dim = 1)   #<! mBBox.shape = (N, 6,  5, 5)
        tB = tB.permute(0, 2, 3, 1).reshape(-1, S * S, 6) #<! mBBox.shape = (N, 25, 6)
		
        return tX, tB

class ToTensor( nn.Module ):
    def __init__( self ) -> None:
        super(ToTensor, self).__init__()
            
    def forward( self: Self, *args ) -> Tuple[torch.Tensor]:
        """
        Converts input to Tensor.  
        """
		
        return tuple(torch.tensor(itm) for itm in args)