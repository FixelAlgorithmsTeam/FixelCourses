# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot Vision Transformer (ViT) Encoder
# Plot kernel like transformation for 1D data.
# Using 1D function one can draw the actual high dimensional kernel on 2D canvas.
# The actual calculation uses `kernel = 'linear'` in order to be able to extract the coefficients.
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
# | 0.1.000 | 22/09/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.datasets import fetch_openml

# Image Processing
import skimage as ski

# Miscellaneous
import os
from platform import python_version, system
import random
# import warnings

# Visualization
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
sampleIdx = 4

# Model
polyDeg    = 15
paramC     = 1e9
kernelType = 'linear'


# Visualization
numGridPts = 500


# %% Loading / Generating Data

mX, vY = fetch_openml('mnist_784', version = 1, return_X_y = True, as_frame = False, parser = 'auto')
vY = vY.astype(np.int_)


# %% Analyze Data

vI = mX[sampleIdx]
mI = np.reshape(vI, (28, 28))

# Generate Grid of 7x7 Patches
mP = ski.util.view_as_windows(mI, (7, 7), step = 7)
mP = np.reshape(mP, (-1, 7, 7))


# %% Plot Results

plt.imshow(mI, cmap = 'gray')

ski.io.imsave('TMP.png', mI.astype(np.uint8))

for ii in range(mP.shape[0]):
    ski.io.imsave(f'TMP{ii:03d}.png', mP[ii].astype(np.uint8))


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

# hF.savefig('TMP.svg', transparent = True)


# %%
