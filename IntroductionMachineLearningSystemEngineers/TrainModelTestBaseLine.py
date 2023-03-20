# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io/)
# 
# # Test Case - Train a Model
# This notebooks follows the test case of object detection.
# This notebooks evaluates the performance of the model before transfer learning.
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
from torchvision import models, transforms

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


# %% Parameters

# Data
imgFolder           = 'Images'
annotationsFolder   = 'Annotations'

imgFolderPath           = os.path.join(DATA_FOLDER_NAME, imgFolder)
annotationsFolderPath   = os.path.join(DATA_FOLDER_NAME, annotationsFolder)

# Model
modelType = MODEL_TYPE_MOBILENET


# Visualization
numRows = 3
numCols = 3


# %% Loading Data (RAW)

lImgFolder = os.listdir(imgFolderPath)
lImg = []
for imgFileName in lImgFolder:
    fileFullPath = os.path.join(imgFolderPath, imgFileName)
    if os.path.isfile(fileFullPath):
        fileName, fileExt = os.path.splitext(imgFileName)
        if fileExt == '.png':
            mI = imread(fileFullPath)
            lImg.append(mI[:, :, :3]) #<! Remove Alpha channel (Some images have it)

lAnnFolder = os.listdir(annotationsFolderPath)
lBox = []

for imgFileName in lAnnFolder:
    fileFullPath = os.path.join(annotationsFolderPath, imgFileName)
    if os.path.isfile(fileFullPath):
        fileName, fileExt = os.path.splitext(imgFileName)
        if fileExt == '.xml':
            lBox.append(ExtractBoxXml(fileFullPath))
                        





# %% Analyze Data

# Plot Data Samples
hF = PlotImagesBBox(lImg, lBox, numRows = numRows, numCols = numCols)
# plt.show()

# Plot Data Distribution
# hF, hA = plt.subplots(figsize = (8, 6))
# hA = PlotLabelsHistogram(dsY, hA = hA)
# plt.show()


# %% Base Model

if modelType == MODEL_TYPE_RESNET_50:
    modelWeights  = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    modelDetector = models.detection.fasterrcnn_resnet50_fpn_v2(weights = modelWeights)
elif modelType == MODEL_TYPE_MOBILENET:
    modelWeights  = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    modelDetector = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights = modelWeights)
elif modelType == MODEL_TYPE_FCOS_RESNET50:
    modelWeights  = models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT
    modelDetector = models.detection.fcos_resnet50_fpn(weights = modelWeights)
elif modelType == MODEL_TYPE_RETINANET:
    modelWeights  = models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
    modelDetector = models.detection.retinanet_resnet50_fpn_v2(weights = modelWeights)
elif modelType == MODEL_TYPE_SSD_LITE:
    modelWeights  = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    modelDetector = models.detection.ssdlite320_mobilenet_v3_large(weights = modelWeights)

# 
modelDetector.eval() #<! Inference mode

# %% Model 

# Infer the Baseline

lBoxPred = []

startTime = time.monotonic()
for ii, mI in enumerate(lImg):
    print(f'Processing image {ii:03d}')
    tI = transforms.functional.to_tensor(mI)
    lRes = modelDetector([tI])
    lBoatBox = [xx.detach().numpy().tolist() for ii, xx in enumerate(lRes[0]['boxes']) if lRes[0]['labels'][ii] == COCOCATEGORY_BOAT_IDX]
    lBoxPred.append(lBoatBox)

endTime = time.monotonic()

print(f'Total run time of inference: {endTime - startTime}')



# %% Plot Results

lPredImg = []
lPredBox = []

for ii, itemBox in enumerate(lBoxPred):
    if len(itemBox) > 0:
        lPredImg.append(lImg[ii])
        lPredBox.append(itemBox)

print(f'Number of detected cases: {len(lPredImg)}')

hF = PlotImagesBBox(lImg, lBoxPred, numRows = 4, numCols = 4)

hF = PlotImagesBBox(lPredImg, lPredBox, numRows = 4, numCols = 4)


# %%
