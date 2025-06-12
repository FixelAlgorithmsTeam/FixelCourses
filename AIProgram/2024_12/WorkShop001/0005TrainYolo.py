# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Object Detection Workshop
# Trains a YOLO v8 Model.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 09/06/2025 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning
from ultralytics import YOLO
from ultralytics import settings

# Miscellaneous
import datetime
import os
from platform import python_version
import random
import warnings
import yaml


# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# Matplotlib default color palette
lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# sns.set_theme() #>! Apply SeaBorn theme


# %% Constants

FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2

DATA_FOLDER_NAME  = 'Data'
TEST_FOLDER_NAME  = 'Test'
TRAIN_FOLDER_NAME = 'Train'
VAL_FOLDER_NAME   = 'Validation'
TILES_FOLDER_NAME = 'Tiles'
IMG_FOLDER_NAME   = 'Images'
LBL_FOLDER_NAME   = 'Labels'
DRIVE_FOLDER_URL  = 'https://drive.google.com/drive/u/2/folders/1wxKIDN777K8kQ4UhJMu5csSbTVXhG7G9'

D_CLS = {'Ball': 0, 'Referee': 1}
L_CLS = ['Ball', 'Referee']


# %% Local Packages


# %% Parameters

modelCfgFile        = 'yolo11n.yaml' #<! The name postfix (`n`) sets the scale of the model
modelWeightsFile    = 'yolo11n.pt' #<! Download from GitHub
dataFile            = 'DetectBall.yaml'

numEpoch    = 100
batchSize   = 12
imgSize     = 640
numWorkers  = 0
ampMode     = False


# %% Ultralytics Settings

# Run once and restart the kernel to apply the changes
# settings.update({
#     'datasets_dir': os.getcwd(), #<! Set the datasets directory to the current directory
#     'sync': False, #<! Disable synchronization with the cloud
#     'clearml': False, #<! Disable ClearML integration
#     'comet': False, #<! Disable Comet integration
#     'hub': False, #<! Disable Ultralytics Hub integration
#     'mlflow': False, #<! Disable MLflow integration
#     'neptune': False, #<! Disable Neptune integration
#     'raytune': False, #<! Disable Ray Tune integration
#     'tensorboard': False, #<! Disable TensorBoard integration
#     'wandb': False, #<! Disable Weights & Biases integration
#     'openvino_msg': False, #<! Disable OpenVINO integration
# })


# %% Define Model

# modelYolo = YOLO(modelCfgFile, cfg = dUltraSettings) #<! Doesn't work
modelYolo = YOLO(modelCfgFile, task = 'detect')
modelYolo = modelYolo.load(modelWeightsFile)


# %% Train the Model

# Edit `settings.yaml` in `AppData\Roaming\Ultralytics`:
# 1. Disable logging.
# 2. Set `datasets_dir: .`.
modelResults = modelYolo.train(data = dataFile, epochs = numEpoch, batch = batchSize, imgsz = imgSize, workers = numWorkers, name = 'DetectionRun', amp = ampMode)


# %% Validate the Model

modelYolo.val()


# %%
