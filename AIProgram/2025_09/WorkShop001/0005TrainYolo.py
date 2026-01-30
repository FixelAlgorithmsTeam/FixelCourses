# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Object Detection Workshop
# Trains a YOLO v8 Model with Hyper Parameter Tuning for Object Detection with Weights & Biases Integration:
# 1. Update the `yolo11n.yaml` file with the number of classes.
# 2. Update the `DetectBall.yaml` file with the paths to the dataset and classes names.
# 3. Update the UltraLytics settings in `settings.yaml` file.
# 4. Build the Sweep configuration.
# 5. Set the Sweep and get the Sweep ID.
# 6. Run the Sweep. Distribute it over multiple computers if available.
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

# ML Ops
import wandb

# Miscellaneous
import os
from platform import python_version
import random
# import warnings

# Typing
from typing import Callable, Dict, Generator, List, Literal, Optional, Self, Set, Tuple, Union

# Visualization
import matplotlib.pyplot as plt

# Jupyter
from IPython import get_ipython


# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

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

PROJECT_NAME     = 'FixelCourses'
DATA_FOLDER_NAME = 'DataSets'
BASE_FOLDER_PATH = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]
DATA_FOLDER_PATH = os.path.join(BASE_FOLDER_PATH, DATA_FOLDER_NAME)

SUB_PROJECT_FOLDER_NAME  = 'BallRefereeDetection'

TEST_FOLDER_NAME  = 'Test'
TRAIN_FOLDER_NAME = 'Train'
VAL_FOLDER_NAME   = 'Validation'
TILES_FOLDER_NAME = 'Tiles'
IMG_FOLDER_NAME   = 'Images'
LBL_FOLDER_NAME   = 'Labels'
DRIVE_FOLDER_URL  = 'https://drive.google.com/drive/u/2/folders/1wxKIDN777K8kQ4UhJMu5csSbTVXhG7G9'

D_CLS = {'Ball': 0, 'Referee': 1}
L_CLS = ['Ball', 'Referee']

ENV_FILE_NAME = '.env'

WANDB_API_KEY      = 'WANDB_API_KEY'
WANDB_ENTITY       = 'WANDB_ENTITY'
WANDB_PROJECT_NAME = 'WANDB_PROJECT_NAME'
WANDB_SWEEP_ID     = 'WANDB_SWEEP_ID'


# %% Local Packages


# %% Auxiliary Functions

def ParseEnvFile( fileName: str = '.env', *, filePath: str = '.', keyValSep: str = '=' ) -> Dict[str, str]:
    # Read the file and parse it into a dictionary
    dEnv = {}
    with open(os.path.join(filePath, fileName), 'r') as hF:
        for line in hF:
            if line.startswith('#') or not line.strip():
                continue
            key, value = line.strip().split(keyValSep, 1)
            dEnv[key]  = value.strip()
    return dEnv

def TrainYoloModel(projName: str, entityName: str, dataFile: str, numEpoch: int, batchSize: int, imgSize: int, numWorkers: int, ampMode: bool, modelWeightsFile: str) -> None:
    """
    Train a YOLO model with hyperparameter tuning using Weights & Biases.  
    Run a single experiment within the sweep configuration defined in Weights & Biases.
    This function initializes a Weights & Biases run, loads the YOLO model, and trains it with the specified parameters.
    Input:
        projName (str): The name of the project in Weights & Biases.
        entityName (str): The name of the entity the project belong to in Weights & Biases.
        dataFile (str): Path to the data configuration file.
        numEpoch (int): Number of epochs to train the model.
        batchSize (int): Batch size for training.
        imgSize (int): Image size for training.
        numWorkers (int): Number of workers for data loading.
        ampMode (bool): Whether to use automatic mixed precision.
        modelWeightsFile (str): Path to the model weights file.
    Output:
        None
    Remarks:
     - the YOLO model, by default, use `optimizer = 'auto'` which overrides `lr0` and `momentum`.
    """
    with wandb.init(entity = entityName, project = projName) as oRun:
        # A single run (Experiment) in the sweep
        dCfg = oRun.config  # Sweep parameters

        # Load model each time to reset weights
        oModel = YOLO(modelWeightsFile)

        # Run training using fixed and sweep defined parameters
        # Hyper parameters by https://docs.ultralytics.com/usage/cfg
        dResults = oModel.train(
            data         = dataFile,
            epochs       = numEpoch,
            imgsz        = imgSize,
            batch        = batchSize,
            workers      = numWorkers,
            amp          = ampMode,
            # project      = projName,
            name         = f'Sweep_{oRun.id}',
            val          = True,
            plots        = True, 
            # Optimizer parameters
            optimizer    = 'auto',               #<! Use 'auto' to let YOLO choose the optimizer, or specify 'SGD', 'Adam', etc.
            # lr0          = dCfg['lr0'],          #<! Have no effect when `optimizer = 'auto'`
            lrf          = dCfg['lrf'],
            # momentum     = dCfg['momentum'],     #<! Have no effect when `optimizer = 'auto'`
            weight_decay = dCfg['weight_decay'], 
            # Augmentation parameters
            hsv_h        = dCfg['hsv_h'],
            hsv_s        = dCfg['hsv_s'],
            hsv_v        = dCfg['hsv_v'],
            degrees      = dCfg['degrees'],
            translate    = dCfg['translate'],
            scale        = dCfg['scale'],
            shear        = dCfg['shear'],        #<!
            perspective  = dCfg['perspective'],  #<!
            flipud       = dCfg['flipud'],
            fliplr       = dCfg['fliplr'],
            bgr          = dCfg['bgr'],
            mosaic       = dCfg['mosaic'],
            mixup        = dCfg['mixup'],
            cutmix       = dCfg['cutmix'],
            erasing      = dCfg['erasing'],      #<! Classification only
        )

        # Validation metrics
        dValMetrics = oModel.val()
        wandb.log({
            'Validation/mAP50-95': float(dValMetrics.box.map),
            'Validation/mAP50': float(dValMetrics.box.map50),
            'Validation/mAP75': float(dValMetrics.box.map75),
            'Validation/mAP5095-Ball': float(dValMetrics.box.maps[0]),
            'Validation/mAP5095-Referee': float(dValMetrics.box.maps[1]),
            'Score': float(dValMetrics.box.map75),
        })


# %% Parameters

modelCfgFile     = 'yolo11n.yaml'           #<! The name postfix (`n`) sets the scale of the model
modelWeightsFile = 'yolo11n.pt'             #<! Download from GitHub
dataFile         = 'DetectBallReferee.yaml' #<! YOLO dataset configuration

# Sweep Configuration
projName   = 'BallRefereeDetection' #<! Project name in Weights & Biases
entityName = 'royia'
numExp     = 10 #<! Number of experiments (Runs) in the sweep

# YOLO Training Parameters (Not in the sweep)
numEpoch    = 15    #<! Adjust according to time budget and complexity of the task
batchSize   = 12    #< Adjust by the GPU memory available
imgSize     = 640   #<! YOLO's default image size
numWorkers  = 2
ampMode     = False #<! Setting to `True` might lower the run time, but might affect the accuracy.


# %% Ultralytics Settings

# Run once and restart the kernel to apply the changes
# settings.update({
#     'datasets_dir': DATA_FOLDER_PATH, #<! Set the datasets directory to the current directory
#     'sync': False, #<! Disable synchronization with the cloud
#     'clearml': False, #<! Disable ClearML integration
#     'comet': False, #<! Disable Comet integration
#     'dvc': False, #<! Disable DVC integration
#     'hub': False, #<! Disable Ultralytics Hub integration
#     'mlflow': False, #<! Disable MLflow integration
#     'neptune': False, #<! Disable Neptune integration
#     'raytune': False, #<! Disable Ray Tune integration
#     'tensorboard': False, #<! Disable TensorBoard integration
#     'wandb': False, #<! Disable Weights & Biases integration
#     'vscode_msg': False, #<! Disable VS Code integration
#     'openvino_msg': False, #<! Disable OpenVINO integration
# })


# %% Configure Weights & Biases

# When working on distributed systems, sharing sensitive data (API Keys, Passwords, etc.) requires caution.
# One way to handle this is to use an environment file (`.env`) that is not shared with the public.
# This section assumes a `.env` file exists in the current directory with the following content:
# ```
# WANDB_API_KEY=<WandB API Key>
# ```
# See example at https://pastebin.com/z3K4TdJa.
# The API Key of the user can be obtained from the Weights & Biases website: https://wandb.ai/authorize.

# Parse the Environment File
dEnv        = ParseEnvFile('.env')
wandbApiKey = dEnv[WANDB_API_KEY] #<! Extract the API Key

entityName = dEnv.get(WANDB_ENTITY, entityName)     #<! Entity name in Weights & Biases
projName   = dEnv.get(WANDB_PROJECT_NAME, projName) #<! Project name in Weights & Biases
sweepId    = dEnv.get(WANDB_SWEEP_ID, None)         #<! Sweep ID in Weights & Biases

wandb.login(key = wandbApiKey, verify = True) #<! Do once per computer


# %% Configure the Sweep

# Based on:
# - https://docs.ultralytics.com/usage/cfg
# - https://docs.ultralytics.com/usage/cfg/#augmentation-settings
# - https://docs.wandb.ai/guides/sweeps/sweep-config-keys
# The choices are mostly to show the options. Review them to make better informed choices.
dSweep = {
    'method': 'random',  # or 'random', 'bayes'
    'metric': {
        'goal': 'maximize',
        'name': 'Score' # Example metric, check YOLO's output for exact name
    },
    'parameters': {
        # 'lr0': {
        #     'distribution': 'log_uniform_values',
        #     'min': 1e-4,
        #     'max': 1e-1
        # },
        'lrf': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 0.2
        },
        # 'momentum': {
        #     'distribution': 'uniform',
        #     'min': 0.85,
        #     'max': 0.97
        # },
        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },
        'hsv_h': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.025
        },
        'hsv_s': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'hsv_v': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'degrees': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3
        },
        'translate': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.3
        },
        'scale': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'shear': {
            'values': [0.0, 10.0],
        },
        'perspective': {
            'values': [0.0, 0.001],
        },
        'flipud': {
            'values': [0.0],
        },
        'fliplr': {
            'values': [0.0, 0.25],
        },
        'bgr': {
            'values': [0.0],
        },
        'mosaic': {
            'values': [0.0],
        },
        'mosaic': {
            'values': [0.0],
        },
        'mixup': {
            'values': [0.0],
        },
        'cutmix': {
            'values': [0.0],
        },
        'erasing': {
            'values': [0.0],
        },
    }
}


# %% Create the Sweep

# Run once to create the sweep
# Then use the sweepId to run the sweep in a distributed manner
# Each node (Computer) will run a different set of experiments
if sweepId is None:
    sweepId = wandb.sweep(dSweep, project = projName)
print(f'Sweep ID: {sweepId}') #>! Print the Sweep ID to use it later

dEnv[WANDB_ENTITY]       = entityName #<! The entity which established teh project
dEnv[WANDB_PROJECT_NAME] = projName
dEnv[WANDB_SWEEP_ID]     = sweepId

envFilePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), ENV_FILE_NAME)

with open(envFilePath, 'w+') as hF:
    for key, val in dEnv.items():
        hF.write(f'{key}={val}\n')


# %% Run the Sweep

# %% Training Function

hTrainYoloModel = lambda: TrainYoloModel(projName, entityName, dataFile, numEpoch, batchSize, imgSize, numWorkers, ampMode, modelWeightsFile)

# On Windows working with the Interactive Mode generates issues with the agent.
# Since the agents uses Multi Processing, the forking causes issues on Windows.
# Hence once the sweep is properly configured, use `0005TrainYoloCLI.py`.

# !!! Run this in CLI. See `0005TrainYoloCLI.py` !!!
# Copy the Sweep ID and use it in `0005TrainYoloCLI.py` to run the sweep.
wandb.agent(sweepId, function = hTrainYoloModel, entity = entityName, project = projName, count = 1) #>! count = number of runs to perform


# %% Validate the Model


# %%
