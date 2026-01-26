# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Shift Notebook Numbering
# Shift the numbers of notebooks.
# 
# > Notebook by:
# > - Royi Avital RoyiAvital@fixelalgorithms.com
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                                         |
# |---------|------------|-------------|------------------------------------------------------------------------------------------|
# | 0.1.000 | 26/01/2026 | Royi Avital | First version                                                                            |
# |         |            |             |                                                                                          |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Image Processing & Computer Vision

# Machine Learning

# Deep Learning

# Miscellaneous
import os
from platform import python_version
import random

# Visualization
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


# %% Local Packages


# %% Auxiliary Functions

from AuxFun import ShiftIndexedFilenames


# %% Parameters


# %% Shift Indexed Filenames

inDir      = r'AIProgram\TMP'
outDir     = r'AIProgram\TMP'
startIndex = 17
endIndex   = 90
valShift   = 64

ShiftIndexedFilenames(inDir, outDir, startIndex, endIndex, valShift)


# %%

