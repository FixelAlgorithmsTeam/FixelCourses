# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - LDA on MNIST Data
# Compares LDA and PCA on MNIST.
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
# | 0.1.000 | 04/03/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

figIdx = 0

# %% Constants


# %% Project Packages


# %% Auxiliary Functions

def IsValidDigit( valY: float, lDigits: List[int] ) -> bool:

    return valY in lDigits


# %% Parameters

# Data
numSamples = 200
dataDim    = 2
mC         = np.array([[0, 1.25], [0, -1.25]])

# Model
numComp = 1


# Visualization
numRows = 3
numCols = 3


# %% Loading / Generating Data

mX, vY = datasets.make_blobs(n_samples = numSamples, n_features = dataDim, centers = mC, cluster_std = [[3, 0.75], [3, 0.75]])


# %% Analyze Data

oPca  = PCA(n_components = numComp)
mXPca = oPca.fit(mX).transform(mX)

oLda  = LinearDiscriminantAnalysis(n_components = numComp)
mXLda = oLda.fit(mX, vY).transform(mX)


# %% Plot Results

# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
sns.set_theme(style = 'ticks', context = 'talk')
plt.style.use('dark_background')  # inverts colors to dark theme

dfData = pd.DataFrame({'x_1': mX[:, 0], 'x_2': mX[:, 1], 'Label': vY})

figIdx += 1

# hF, hA = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 6))
hJointGrid = sns.jointplot(data = dfData, x = 'x_1', y = 'x_2', hue = 'Label',
                           xlim = (-10, 10), ylim = (-10, 10), 
                           palette = 'tab10')
hF = hJointGrid.figure
hJointGrid.set_axis_labels('$x_1$', '$x_2$')


# hA.set_title('Data')
# # sns.scatterplot(data = dfDataLda, x = 'x_1', y = 'x_2', hue = 'Label', 
# #                 palette = 'tab10', ax = vHa[1])
# hA.set_title('LDA')
# hA.set_xlabel('$x_1$')
# hA.set_ylabel('$x_2$')

hF.savefig(f'Figure{figIdx:04d}.svg', transparent = True)


# %%
