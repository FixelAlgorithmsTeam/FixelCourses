# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot Diffusion Models
# Visualizations for the Diffusion Models slides.
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
# | 0.1.000 | 05/09/2026 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning

# Image Processing
import imageio.v3 as iio

# Miscellaneous
import os
from platform import python_version, system
import random
# import warnings

# Visualization
from matplotlib import patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns

# Typing
from typing import Callable, List, Tuple, Union

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants


# %% Courses Packages


# %% Auxiliary Functions



# %% Parameters

# Data
owlImgUrl = r'https://i.imgur.com/oDkhajH.png' #<! From https://www.flickr.com/photos/grovecrest/18642489553 + 110 Saturation in Paint.net
owlImgUrl = r'https://i.postimg.cc/dVg23f79/18642489553-cff4338cb8-o.png' #<! From https://www.flickr.com/photos/grovecrest/18642489553 + 110 Saturation in Paint.net


# Model
numGridPtsT = 5
noiseStd    = 0.40

lNoiseStd = [0.15, 0.50, 0.95, 1.15, 2.00]


# Visualization
numGridPts = 500


# %% Noise Evolution

tI = iio.imread(owlImgUrl) / 255.0

for t in range(numGridPtsT):
    tN = np.random.normal(0, lNoiseStd[t], tI.shape)
    tI = np.clip(tI + tN, 0, 1)
    plt.imshow(tI)
    plt.title(f'Noise Evolution - Step {t+1}')
    plt.axis('off')
    plt.show()
    iio.imwrite(f'NoiseEvolution_Step{t+1}.png', np.round(tI * 255).astype(np.uint8))


# %% Analyze Data





# %% Plot Results

# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
# sns.set_theme(style = "ticks", context = "talk")
# plt.style.use("dark_background")  # inverts colors to dark theme

# hF, hA = plt.subplots(figsize = (10, 6))
# sns.scatterplot(x = mXP[:, 0], y = mXP[:, 1], hue = vY,
#                 palette = 'tab10', ax = hA)
# hA.set_title('Kernel Trick')
# sns.lineplot(x = vX, y = vC, 
#              ax = hA, color = 'oldlace', label = 'Decision Boundary')
# hA.set_xlabel('$x_1$')
# hA.set_ylabel('$x_2$');

# # hF.savefig('TMP.svg', transparent = True)


# %%
