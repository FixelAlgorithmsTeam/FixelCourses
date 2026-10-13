# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
#
# # Process Files
# Process files using the `TimeExtractor` class.
# Remarks:
# - Should not be used for the new data structure with sub folders for the scene type.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 1.0.000 | 08/04/2026 | Royi Avital | First version                                                                            |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Typing
from typing import Any, Callable, Dict, Generator, List, Optional, Self, Set, Tuple, Union
from numpy.typing import NDArray

# Image Processing & Computer Vision
import skimage as ski

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random
import time

# Visualization
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

L_IMG_EXT = ['.png', '.jpeg', '.jpg']

PROJECT_NAME        = 'KickAI'
DATA_FOLDER         = 'Data'
SUB_PROJECT_NAME    = 'TimeExtraction' #<! Matching data folder name in `Data` folder

PROJECT_BASE_FOLDER = os.getcwd()[:(len(os.getcwd()) - (os.getcwd()[::-1].lower().find(PROJECT_NAME.lower()[::-1])))]


# %% Local Packages

from LLMSession import ModelSupplier, TimeExtractor, TimeExtractorGemini, TimeExtractionResult


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


# %% Parameters

# modelName    = 'gpt-5.4-mini'
# modelName    = 'gpt-5.4-nano'
modelName    = 'gemini-3-flash-preview'
# modelName    = 'gemini-3.1-flash-lite-preview'

imgName      = '19934641_6346.png'
resizeFctr   = 0.5
tuCropSize   = (0, 0, 700, 300) #<! (x, y, w, h)

resAppnedix  = '_Result'

match modelName:
    case 'gpt-5.4' | 'gpt-5.4-pro' | 'gpt-5.4-mini' | 'gpt-5.4-nano':
        keyFieldName = 'OPENAI_API_KEY'
    case 'gemini-3.1-pro-preview' | 'gemini-3-flash-preview' | 'gemini-3.1-flash-lite-preview':
        keyFieldName = 'GEMINI_API_KEY'
    case _:
        raise ValueError(f'Unsupported model name: {modelName!r}')

fontSize = 24


# %% Load / Generate Data

mI = ski.io.imread(os.path.join(PROJECT_BASE_FOLDER, DATA_FOLDER, SUB_PROJECT_NAME, imgName))

dEnv   = ParseEnvFile('.env')
apiKey = dEnv[keyFieldName]


# %%

oTimeExt = TimeExtractor(apiKey, modelName, resizeFctr = resizeFctr, tuCropSize = tuCropSize)
# oTimeExt = TimeExtractorGemini(apiKey, modelName, resizeFctr = resizeFctr, tuCropSize = tuCropSize)


# %%

# Measure Run Time
startTime = time.time()
timeExtResult = oTimeExt.Predict(mI)
endTime = time.time()
print(f'Time extraction took {endTime - startTime:.2f} [Sec].')

if timeExtResult.hasTime:
    print(f'Extracted time: {timeExtResult.matchTime} with certainty of {timeExtResult.valConfidence:.2f}.')
    print(f'The 1st team score is {timeExtResult.firstTeamScore} and the 2nd team score is {timeExtResult.secondTeamScore}.')
else:
    print(f'No time extracted. Certainty of no time is {timeExtResult.valConfidence:.2f}.')


# %% Process the Folder

os.listdir(os.path.join(PROJECT_BASE_FOLDER, DATA_FOLDER, SUB_PROJECT_NAME))
# Filter Images
lImages = [f for f in os.listdir(os.path.join(PROJECT_BASE_FOLDER, DATA_FOLDER, SUB_PROJECT_NAME)) if os.path.splitext(f)[1].lower() in L_IMG_EXT]  
# Images should not include the `resAppnedix` in their name to avoid processing already processed images
lImages = [f for f in lImages if resAppnedix not in os.path.splitext(f)[0]]
lImages.sort() #<! Sort images by name

for imgName in lImages:
    mI = ski.io.imread(os.path.join(PROJECT_BASE_FOLDER, DATA_FOLDER, SUB_PROJECT_NAME, imgName))
    startTime = time.time()
    timeExtResult = oTimeExt.Predict(mI)
    endTime = time.time()
    print(f'Image: {imgName} - Extracted time: {timeExtResult.matchTime} with certainty of {timeExtResult.valConfidence:.2f}.')
    print(f'Time extraction took {endTime - startTime:.2f} [Sec].')

    # Generate Image with Extracted Time Overlay
    hF = plt.figure()
    hF.figimage(mI, resize = True)
    hF.text(0.01, 0.40, f'Extracted Time: {"True" if timeExtResult.hasTime else "False"}', fontsize = fontSize, color = 'red')
    if timeExtResult.hasTime:
        hF.text(0.01, 0.35, f'Extracted Time: {timeExtResult.matchTime}', fontsize = fontSize, color = 'red')
    if timeExtResult.stoppageTimeReferee is not None:
        hF.text(0.01, 0.30, f'Referee Stoppage Time: {timeExtResult.stoppageTimeReferee}', fontsize = fontSize, color = 'red')
    if timeExtResult.stoppageTimeActual is not None:
        hF.text(0.01, 0.25, f'Actual Stoppage Time: {timeExtResult.stoppageTimeActual}', fontsize = fontSize, color = 'red')
    if timeExtResult.firstTeamScore is not None:
        hF.text(0.01, 0.20, f'First Team Score: {timeExtResult.firstTeamScore}', fontsize = fontSize, color = 'red')
    if timeExtResult.secondTeamScore is not None:
        hF.text(0.01, 0.15, f'Second Team Score: {timeExtResult.secondTeamScore}', fontsize = fontSize, color = 'red')
    hF.text(0.01, 0.10, f'Run Time: {endTime - startTime:.2f} [Sec]', fontsize = fontSize, color = 'red')
    hF.canvas.draw()
    mA = np.asarray(hF.canvas.renderer.buffer_rgba())
    mA = ski.transform.resize(mA, (540, 960))
    mA = ski.util.img_as_ubyte(mA)
    plt.close(hF)
    ski.io.imsave(os.path.join(PROJECT_BASE_FOLDER, DATA_FOLDER, SUB_PROJECT_NAME, f'{os.path.splitext(imgName)[0] + resAppnedix}.png'), mA)


# %%
