# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Object Detection Workshop
# Infers using a YOLO v8 Model.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 07/07/2024 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random

# Visualization
import matplotlib as mpl
from matplotlib.patches import Rectangle, Circle
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

from AuxFun import BBoxFormat
from AuxFun import ConvertBBoxFormat, PlotBox, PlotBBox
from TiledYoloDetector import TiledDetector #<! Supports tiling detection

# %% Parameters

imgFileExt  = 'png'

# Should be the path to the model file.
# Extract it from the optimal sweep folder.
# modelPath = ???
modelPath = r'D:\Applications\Documents\FixelAlgorithms\Courses\FixelCourses\AIProgram\2025_09\WorkShop001\runs\detect\Sweep_92xbm5j6\weights\best.pt'

# Set parameters to load pre trained model


# %% Define Model

oTileDet = TiledDetector(modelFilePath = modelPath, confThr = 0.25)


# %% Inference Using the Model

dataFolderPath  = os.path.join(DATA_FOLDER_PATH, SUB_PROJECT_FOLDER_NAME) #<! Path to folder with images and labels (LabelMe JSON Files)

# Show a single image with its labels
lFile    = [fileName for fileName in os.listdir(dataFolderPath) if fileName.endswith(imgFileExt)] #<! Assuming all images are labeled
numFiles = len(lFile)

# %%

imgIdx = random.randrange(numFiles)
# imgIdx = 224 #<! Used in the slides

imgFullName          = lFile[imgIdx]
fileName, imgFileExt = os.path.splitext(imgFullName)

# Read image
mI = ski.io.imread(os.path.join(dataFolderPath, imgFullName))
tuImgSize = (mI.shape[0], mI.shape[1]) #<! (numRows, numCols) = (imgHeight, imgWidth)
# Return Boxes and Confidence levels
# The boxes are Pascal VOC format: `(xmin, ymin, xmax, ymax)`
mB, vConf = oTileDet.Predict(mI)

mB = mB[vConf > 0] #<! Keep only detected boxes (Classes)
for ii in range(mB.shape[0]):
    mB[ii] = ConvertBBoxFormat(mB[ii], tuImgSize, BBoxFormat.PASCAL_VOC, BBoxFormat.YOLO)
vLbl = np.arange(mB.shape[0])
lLbl = [oTileDet.GetLabelName(ii) for ii in vLbl]

hA = PlotBox(mI, vLbl, mB, lLabelText = lLbl)
hA.set_title(f'Image: {imgIdx}')
hA.legend();


# %%
