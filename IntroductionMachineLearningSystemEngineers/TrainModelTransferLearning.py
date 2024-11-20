# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Test Case - Train a Model
# This notebooks follows the test case of object detection.
# This notebooks uses a transfer learning approach to train a model.
#
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks:
# - A
# 
# To Do & Ideas:
# 1. B
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 0.1.000 | 17/03/2023 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
# from sklearn.model_selection import train_test_split

# PyTorch
# PyTorch
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from torchview import draw_graph


# Image Processing
from skimage.io import imread
# from skimage.color import rgba2rgb

# Miscellaneous
# from collections import OrderedDict
import datetime
import json
import os
from platform import python_version, system
import random
import time
import xml.etree.ElementTree as ET
# import warnings
import yaml


# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

PROJECT_DIR_NAME = 'FixelCourses'
PROJECT_DIR_PATH = os.path.join(os.getcwd()[:os.getcwd().find(PROJECT_DIR_NAME)], PROJECT_DIR_NAME)

DATA_FOLDER_NAME    = os.path.join(PROJECT_DIR_PATH, 'DataSets', 'DetectShips')
DATA_FILE_NAME      = 'shipsnet.json'
CSV_FILE_NAME       = 'ImgData.csv'
# MODEL_FOLDER_NAME   = 'Model'
# FIG_FOLDER_NAME     = 'Figures'

MODEL_TYPE_RESNET_50        = 1 #<! 1400 Sec, 255 Cases
MODEL_TYPE_MOBILENET        = 2 #<! 60 Sec, 67 Cases
MODEL_TYPE_FCOS_RESNET50    = 3 #<! 1000 Sec, 304 Cases (Missed some small ones)
MODEL_TYPE_RETINANET        = 4 #<! 919 Sec, 271 Cases (Missed some small ones)
MODEL_TYPE_SSD_LITE         = 5 #<! 76 Sec, 223 Cases (Some false alarms)


	
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 
                                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                                'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 
                                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
                                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
                                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
                                'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 
                                'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 
                                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCOCATEGORY_BOAT_IDX = COCO_INSTANCE_CATEGORY_NAMES.index('boat')


# %% Project Packages

from AuxFun import *
from ModelsModule import *
from TrainerModule import *


# %% Parameters

modelName       = 'FAShipDetector'
modelVersion    = '0.1.000'
confFileName    = 'ShipDetectorConf.yaml'

debugMode = True

# Data
imgFolder = 'Images'
annFolder = 'Annotations'

imgFolderPath   = os.path.join(DATA_FOLDER_NAME, imgFolder)
annFolderPath   = os.path.join(DATA_FOLDER_NAME, annFolder)
# dataSetName     = 'Dataset7_FSD'
trainSize       = 0.9
batchSize       = 5
shuffleEpoch    = False

# Data Augmentation
dataAug = None #<! Default, Change it in the dedicated cell

# Net Model
modelType = MODEL_TYPE_MOBILENET
numClass  = 2 #<! Background + Ship

# Trainer
numEpochs           = 20
accelMode           = 'auto'
accumulateBatches   = 10

dOptConf = {
    'optimizer_type': OptimizerType.ADAM,
    'lr': 0.00075
}

dSchedConf = {
    'scheduler_type': SchedulerType.STEP_LR,
    'step_size': 1,
    'gamma': 0.99
}

dSchedConf = {
    'scheduler_type': SchedulerType.LINEAR_LR,
    'start_factor': 1,
    'end_factor': 0.005,
    'total_iters': numEpochs
}

# Debug Mode
debugModeTrainNumSamples    = 50
debugModeValNumSamples      = 5
debugModeNumEpochs          = 20

displayValBatches = 3

tbLogDirBase = 'TensorBoardLog'


# %% Loading Data

hCollaterFun = TorchObjDetectionCollateFn()

dsShipData  = PascalVocLoader(imgFolderPath, annFolderPath)

numSamples = len(dsShipData)

vTrainIdx, vValIdx = GenTrainTesIdx(numSamples, trainSize = trainSize, seedNum = seedNum)

dsTrain = Subset(dsShipData, vTrainIdx)
dsVal   = Subset(dsShipData, vValIdx)

if debugMode:
    dsTrain     = Subset(dsTrain, range(debugModeTrainNumSamples))
    dsVal       = Subset(dsVal, range(debugModeValNumSamples))
    numEpochs   = debugModeNumEpochs

if system() == 'Windows':
    numWorkers = 0 #<! Royi: On Windows num workers > 0 makes things stupidly slower (persistent_workers = True won't work)

# persistent_workers = True

dlTrain = DataLoader(dsTrain, batch_size = batchSize, shuffle = shuffleEpoch, num_workers = numWorkers, collate_fn = hCollaterFun)
dlVal   = DataLoader(dsVal, batch_size = batchSize, shuffle = shuffleEpoch, num_workers = numWorkers, collate_fn = hCollaterFun)


# %% Display Data

lImg, lTarget = next(iter(dlTrain))

lImg = [np.transpose(tI.numpy(), (1, 2, 0)) for tI in lImg]
lBox = [dItem['boxes'].tolist() for dItem in lTarget]

numCols = 6
numRows = int(np.ceil(batchSize / numCols))

hF = PlotImagesBBox(lImg, lBox, numRows = numRows, numCols = numCols)
# plt.show()


# %% Data Augmenter


# %% Model

if modelType == MODEL_TYPE_RESNET_50:
    backBoneWeights = models.ResNet50_Weights.DEFAULT
    modelDetector   = models.detection.fasterrcnn_resnet50_fpn_v2(weights = None, num_classes = numClass, weights_backbone = backBoneWeights)
elif modelType == MODEL_TYPE_MOBILENET:
    backBoneWeights = models.MobileNet_V3_Large_Weights.DEFAULT
    modelDetector   = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = None, num_classes = numClass, weights_backbone = backBoneWeights)
elif modelType == MODEL_TYPE_FCOS_RESNET50:
    backBoneWeights = models.ResNet50_Weights.DEFAULT
    modelDetector   = models.detection.fcos_resnet50_fpn(weights = None, num_classes = numClass, weights_backbone = backBoneWeights)
elif modelType == MODEL_TYPE_RETINANET:
    backBoneWeights = models.ResNet50_Weights.DEFAULT
    modelDetector   = models.detection.retinanet_resnet50_fpn_v2(weights = None, num_classes = numClass, weights_backbone = backBoneWeights)
elif modelType == MODEL_TYPE_SSD_LITE:
    backBoneWeights = models.MobileNet_V3_Large_Weights.DEFAULT
    modelDetector   = models.detection.ssdlite320_mobilenet_v3_large(weights = None, num_classes = numClass, weights_backbone = backBoneWeights)



# %% Trainer (PyTorch Lightning)

tbLogger = TensorBoardLogger(save_dir = tbLogDirBase, log_graph = True)
# modelTrainer = pl.Trainer(max_epochs = numEpochs, accelerator = accelMode, accumulate_grad_batches = accumulateBatches, callbacks = [DeviceStatsMonitor(), LearningRateMonitor(), RichProgressBar()], logger = tbLogger)
modelTrainer = pl.Trainer(max_epochs = numEpochs, accelerator = accelMode, callbacks = [DeviceStatsMonitor(), LearningRateMonitor(), RichProgressBar()], logger = tbLogger)
modelWrapper = LitModel(modelDetector, dOptConf = dOptConf, dSchedConf = dSchedConf)

# %% Training

modelTrainer.fit(modelWrapper, train_dataloaders = dlTrain, val_dataloaders = dlVal)
# tensorboard --logdir lightning_logs

# %% Save Model Parameters (YAML)

chkPtPath, chkPtFileName = os.path.split(modelTrainer.checkpoint_callback.best_model_path)

dModelYaml = {}
dModelYaml['modelName']     = modelName
dModelYaml['modelVersion']  = modelVersion
dModelYaml['modelFile']     = chkPtFileName
dModelYaml['modelType'] = modelType
dModelYaml['backBoneWeights'] = str(backBoneWeights)
dModelYaml['numClass']      = numClass
if dataAug is None:
    dModelYaml['transform'] = 'None'
else:
    dModelYaml['transform'] = {'ClassName': dataAug.name, 'Version': dataAug.version}
dModelYaml['runDevice']     = 'cpu'
dModelYaml['optConf']       = dOptConf
dModelYaml['schedConf']     = dSchedConf

# YAML friendly types
dModelYaml['optConf']['optimizer_type']     = dModelYaml['optConf']['optimizer_type'].name
dModelYaml['schedConf']['scheduler_type']   = dModelYaml['schedConf']['scheduler_type'].name

# Save data
hFile = open(os.path.join(modelTrainer.log_dir, confFileName), 'w', encoding = 'utf-8')
yaml.safe_dump(dModelYaml, hFile)
hFile.close()

# %% Display Predictions (Validation Data)

# for ii, (lImg, lTarget) in enumerate(dlVal):
    
#     numSamples = len(lImg)
#     lPred = modelTrainer.model.forward(lImg)
    
#     for jj in range(numSamples):
#         inputImg    = lImg[jj].clone().detach().cpu().numpy()
#         lBox        = lTarget[jj]['boxes'].tolist()
#         lBoxPred    = lPred[jj]['boxes'].tolist()
#         hF = PlotNetPred(inputImg, lablesImg, outImg, binClassThr = segThr)
    
#     if (ii + 1) == displayValBatches:
#         break


# %%



