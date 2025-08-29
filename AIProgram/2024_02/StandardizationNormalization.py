# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Machine Learning - Feature Engineering - Standardization vs. Normalization
# Shows the effects of standardization vs. normalization.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
#
# Remarks
# - A
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 09/03/2024 | Royi Avital | First version                                                                            |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision

# Machine Learning
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Deep Learning

# Optimization

# Miscellaneous
import os
from platform import python_version
import random

from typing import Callable, Dict, List, Optional, Set, Tuple, Union


# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Jupyter
from IPython import get_ipython
from IPython.display import Image
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, interact, IntSlider, Layout, SelectionSlider
from ipywidgets import interact

# %% [markdown]
# ## Notations
# 
# * <font color='red'>(**?**)</font> Question to answer interactively.
# * <font color='blue'>(**!**)</font> Simple task to add code for the notebook.
# * <font color='green'>(**@**)</font> Optional / Extra self practice.
# * <font color='brown'>(**#**)</font> Note / Useful resource / Food for thought.
# 
# Code Notations:
# 
# ```python
# someVar    = 2; #<! Notation for a variable
# vVector    = np.random.rand(4) #<! Notation for 1D array
# mMatrix    = np.random.rand(4, 3) #<! Notation for 2D array
# tTensor    = np.random.rand(4, 3, 2, 3) #<! Notation for nD array (Tensor)
# tuTuple    = (1, 2, 3) #<! Notation for a tuple
# lList      = [1, 2, 3] #<! Notation for a list
# dDict      = {1: 3, 2: 2, 3: 1} #<! Notation for a dictionary
# oObj       = MyClass() #<! Notation for an object
# dfData     = pd.DataFrame() #<! Notation for a data frame
# dsData     = pd.Series() #<! Notation for a series
# hObj       = plt.Axes() #<! Notation for an object / handler / function handler
# ```
# 
# ### Code Exercise
# 
#  - Single line fill
# 
#  ```python
#  valToFill = ???
#  ```
# 
#  - Multi Line to Fill (At least one)
# 
#  ```python
#  # You need to start writing
#  ????
#  ```
# 
#  - Section to Fill
# 
# ```python
# #===========================Fill This===========================#
# # 1. Explanation about what to do.
# # !! Remarks to follow / take under consideration.
# mX = ???
# 
# ???
# #===============================================================#
# ```

# %% Configuration

# %matplotlib inline
# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

# Matplotlib default color palette
lMatPltLibclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# sns.set_theme() #>! Apply SeaBorn theme

runInGoogleColab = 'google.colab' in str(get_ipython())

# %% Constants

FIG_SIZE_DEF    = (8, 8)
ELM_SIZE_DEF    = 50
CLASS_COLOR     = ('b', 'r')
EDGE_COLOR      = 'k'
MARKER_SIZE_DEF = 10
LINE_WIDTH_DEF  = 2


# %% Course Packages

from DataVisualization import Plot2DLinearClassifier, PlotBinaryClassData

# %% Auxiliary Functions


# %% Parameters

numPts = 30
tuDimX1 = (20, 100)
tuDimX2 = (-10, 10)


# %% Generate / Load Data

vX1 = np.random.uniform(low = tuDimX1[0], high = tuDimX1[1], size = numPts)
vX2 = np.random.uniform(low = tuDimX2[0], high = tuDimX2[1], size = numPts)

mX = np.c_[vX1, vX2]


# %% Plot Data

# Dark Theme + Transparency
# Works on VS Code by copying the output.
plt.style.use('dark_background')
hF, hA = plt.subplots(figsize = (10, 7))
hF.patch.set_alpha(0.0) #<! Transparent background
hA.patch.set_alpha(0.0) #<! Transparent axes
hA.scatter(mX[:, 0], mX[:, 1], s = 50)
hA.set_aspect('equal')
hA.set_xlabel(r'$x_1$')
hA.set_ylabel(r'$x_2$')
hA.set_title(f'Original Data, Mean: {np.mean(mX, axis = 0)}, Variance: {np.std(mX, axis = 0)}')


# %% Apply Normalization / Standardization


lTrns = [(MinMaxScaler(), 'Normalized'), (StandardScaler(), 'Standardized')]


for ii, (oTrns, trnsStr) in enumerate(lTrns):
    mZ = oTrns.fit_transform(mX)
    hF, hA = plt.subplots(figsize = (10, 7))
    hF.patch.set_alpha(0.0)
    hA.patch.set_alpha(0.0)
    hA.scatter(mZ[:, 0], mZ[:, 1], s = 50)
    hA.set_aspect('equal')
    hA.set_xlabel(r'$x_1$')
    hA.set_ylabel(r'$x_2$')
    hA.set_title(f'{trnsStr} Data, Mean: {np.mean(mZ, axis = 0)}, Variance: {np.std(mZ, axis = 0)}')

# %%
