# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot Kernel Trick
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
# | 0.1.000 | 25/02/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC

# Image Processing

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

# warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants


# %% Courses Packages


# %% Auxiliary Functions


# %% Parameters

# Data
numSamples = 8

# Model
polyDeg    = 15
paramC     = 1e9
kernelType = 'linear'


# Visualization
numGridPts = 500


# %% Loading / Generating Data

mX  = np.linspace(0, 1, numSamples)[:, None]
mXP = np.column_stack((mX, np.zeros(numSamples)))  #<! For Plotting
# vY  = np.random.choice(2, size = numSamples)
vY  = np.zeros(numSamples, dtype = np.int32)
vY[::2] = 1


# %% Analyze Data

oPolyTrns = PolynomialFeatures(degree = polyDeg, include_bias = False)
mXX = oPolyTrns.fit_transform(mX)

oSvcCls = SVC(C = paramC, kernel = kernelType)
oSvcCls = oSvcCls.fit(mXX, vY)

print(oSvcCls.score(mXX, vY)) #<! Increase `paramC` / `polyDeg` until it is 1.0

vW             = oSvcCls.coef_[0]
paramIntercept = oSvcCls.intercept_[0]


vX = np.linspace(0, 1, numGridPts)
mXXP = oPolyTrns.fit_transform(vX[:, None])

vC = mXXP @ vW + paramIntercept

# Alternate calculation using `polyval()`
################################################################################
# mXX = np.pow(mX, np.arange(1, polyDeg + 1)[None, :])

# oSvcCls = SVC(C = paramC, kernel = kernelType)
# oSvcCls = oSvcCls.fit(mXX, vY)

# print(oSvcCls.score(mXX, vY)) #<! Increase `paramC` / `polyDeg` until it is 1.0

# vW             = oSvcCls.coef_[0]
# paramIntercept = oSvcCls.intercept_[0]
# vP = np.r_[vW[::-1], paramIntercept]
# vX = np.linspace(0, 1, numGridPts)
# vC = np.polyval(vP, vX)
################################################################################


# %% Plot Results

# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
sns.set_theme(style = "ticks", context = "talk")
plt.style.use("dark_background")  # inverts colors to dark theme

hF, hA = plt.subplots(figsize = (10, 6))
sns.scatterplot(x = mXP[:, 0], y = mXP[:, 1], hue = vY,
                palette = 'tab10', ax = hA)
hA.set_title('Kernel Trick')
sns.lineplot(x = vX, y = vC, 
             ax = hA, color = 'oldlace', label = 'Decision Boundary')
hA.set_xlabel('$x_1$')
hA.set_ylabel('$x_2$');

# hF.savefig('TMP.svg', transparent = True)


# %%
