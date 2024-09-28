# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Test Case - Train a Model
# This notebooks follows the test case of object detection.
# This notebook trains a small PoC.
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
# | 0.1.000 | 15/03/2023 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
# from sklearn.model_selection import train_test_split

# PyTorch
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.callbacks import Checkpoint, EpochScoring, LRScheduler, TensorBoard
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph


# Image Processing
import PIL 

# Miscellaneous
# from collections import OrderedDict
import datetime
import json
import os
from platform import python_version, system
import random
# import warnings
# import yaml


# Visualization
# from bokeh.plotting import figure, show
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

# %% Project Packages

from AuxFun import *
from ModelsModule import *


# %% Parameters

# Data
tImgSize    = (80, 80)
numChannels = 3
numPixels   = tImgSize[0] * tImgSize[1] * numChannels

# Model
numOutputs = 1 #<! Single output (Probability of the class 1)
lossFun = nn.BCEWithLogitsLoss
clsThr = 0.5 #<! Threshold for class 1
learningRate = 0.001
batchSize  = 50
numEpoch = 25
optimizerFun = torch.optim.SGD
optimizerMomentum = 0.9
shuffleData = True
numWorkes = 0
if system() == 'Windows':
    numWorkers = 0 #<! Royi: On Windows num workers > 0 makes things stupidly slower (persistent_workers = True won't work)
dataSetLoader   = ShipsImgLoader
torchDevice     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tbLogDirBase    = 'TensorBoardLog'


# Visualization
numRows = 3
numCols = 3


# %% Loading Data (RAW)

if os.path.isfile(os.path.join(DATA_FOLDER_NAME, CSV_FILE_NAME)):
    print('Found CSV File')
    dfData = pd.read_csv(os.path.join(DATA_FOLDER_NAME, CSV_FILE_NAME), dtype = int, engine = 'pyarrow')

    dfX = dfData.drop(columns = ['Label'])
    dfX = dfX.astype(np.uint8)
    dsY = dfData['Label']
else:
    hFile = open(os.path.join(DATA_FOLDER_NAME, DATA_FILE_NAME), 'r')
    dJsonData = json.load(hFile)
    hFile.close()
    numFiles = len(dJsonData['data'])
    
    dfX = pd.DataFrame(data = np.zeros(shape = (numFiles, numPixels), dtype = np.uint8))
    dsY = pd.Series(np.zeros(shape = numFiles, dtype = int), name = 'Label') #<! Label per file
    
    for ii in range(numFiles):
        dfX.iloc[ii, :] = dJsonData['data'][ii]
        dsY.iloc[ii]    = dJsonData['labels'][ii]
    
    dfData = pd.concat((dfX, dsY), axis = 1)
    dfData.to_csv(os.path.join(DATA_FOLDER_NAME, CSV_FILE_NAME), index = False)

numSamples = dfX.shape[0]


# %% Analyze Data

# Plot Data Samples
hF = PlotImages(dfX.to_numpy(), dsY.to_numpy(), numRows = numRows, numCols = numCols, tImgSize = tImgSize, numChannels = 3)
# plt.show()

# Plot Data Distribution
hF, hA = plt.subplots(figsize = (8, 6))
hA = PlotLabelsHistogram(dsY, hA = hA)
# plt.show()


# %% Model Graph

modelGraph = draw_graph(NnImageCls(numOutputs), input_size = (batchSize, 3, 80, 80), graph_name = 'PoC', device = 'meta')
modelGraph.visual_graph

# modelGraph = draw_graph(NnImageCls(numOutputs), input_size = (batchSize, 3, 80, 80), 
#                         graph_name = 'PoC', depth = 7, device = 'cuda', strict = False, expand_nested = True, 
#                         hide_inner_tensors = False, save_graph = True)
# modelGraph.visual_graph

# %% Model

currTime    = datetime.datetime.now()
folderName  = currTime.strftime('%Y_%m_%d_%H_%M_%S')
tbLogDir    = os.path.join(tbLogDirBase, folderName)

tbWriter = SummaryWriter(log_dir = tbLogDir)

checkPoint          = Checkpoint(f_params = 'bestModel.pt', monitor = 'valid_acc_best', dirname = tbLogDir)
scoreF1             = EpochScoring(scoring = 'f1', lower_is_better = False, name = 'valid_f1')
lrScheduler         = LRScheduler(policy = 'StepLR', step_size = 7, gamma = 0.1)
tensorBoardLogger   = TensorBoard(tbWriter, close_after_train = True)

netCls = NeuralNetBinaryClassifier(
    NnImageCls(numOutputs), 
    criterion = lossFun,
    threshold = clsThr,
    lr = learningRate,
    batch_size = batchSize,
    max_epochs = numEpoch,
    optimizer = optimizerFun,
    optimizer__momentum = optimizerMomentum,
    iterator_train__shuffle = shuffleData,
    iterator_train__num_workers = numWorkes,
    iterator_valid__num_workers = numWorkes,
    dataset = dataSetLoader,
    callbacks = [lrScheduler, checkPoint, scoreF1, tensorBoardLogger],
    device = torchDevice
)

tuTensorSize = (batchSize, ) + (numChannels, ) + tImgSize
tbWriter.add_graph(model = NnImageCls(numOutputs), input_to_model = torch.rand(size = tuTensorSize))


# %% Train the Model

mX = dfX.to_numpy()
vY = dsY.to_numpy()
vY = vY.astype(np.float32) #<! Labels become probability of Class 1

netCls.fit(mX, vY)
# tensorBoardLogger.add_scalar_maybe(netCls.history, key = 'valid_f1', tag = 'F1 Score')
# tbWriter.close()

# %% Predicted by Model

vYPred = netCls.predict(mX)


# %% Analyze Results

# Performance Graph
hF, hAs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
hAs = hAs.flat

hAs[0].plot(range(numEpoch), netCls.history[:, 'train_loss'], label = 'Train Loss')
hAs[0].plot(range(numEpoch), netCls.history[:, 'valid_loss'], label = 'Validation Loss')
hAs[0].set_xlabel('Epoch Index')
hAs[0].set_ylabel('Loss')
hAs[0].legend()

hAs[1].plot(range(numEpoch), netCls.history[:, 'valid_acc'], label = 'Validation Accuracy')
hAs[1].plot(range(numEpoch), netCls.history[:, 'valid_f1'], label = 'Validation F1')
hAs[1].set_xlabel('Epoch Index')
hAs[1].set_ylabel('Score')
hAs[1].legend()

plt.show()

# Confusion Matrix

hF, hA = plt.subplots(figsize = (7, 7))
PlotConfusionMatrix(vY, vYPred, hA = hA)

plt.show()

# Analyzing the False Cases
vFalseIdx       = np.flatnonzero(vY != vYPred)
numFalseCases   = np.size(vFalseIdx)

if numFalseCases > 0:
    numColsDisp = 6
    numRowsDisp = int(np.ceil(numFalseCases / numColsDisp))

    hF = PlotImages(mX[vFalseIdx], vY[vFalseIdx], numRows = numRowsDisp, numCols = numColsDisp, tImgSize = tImgSize, numChannels = 3)
    # plt.show()


# %%
