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
# | 0.1.000 | 22/02/2025 | Royi Avital | First version                                                      |
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

# warnings.filterwarnings("ignore")

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
lDigits = [0, 1, 3, 7]

# Model
numComp = 2


# Visualization
numRows = 3
numCols = 3


# %% Loading / Generating Data

mX, vY = datasets.load_digits(return_X_y = True)
hIsValidDigit = lambda valY: IsValidDigit(valY, lDigits)
hIsValidDigitVec = np.vectorize(hIsValidDigit)
vIdx = hIsValidDigitVec(vY)

vY = vY[vIdx]
mX = mX[vIdx]


# %% Analyze Data

oPca  = PCA(n_components = numComp)
mXPca = oPca.fit(mX).transform(mX)

oLda  = LinearDiscriminantAnalysis(n_components = numComp)
mXLda = oLda.fit(mX, vY).transform(mX)

dfDataPca = pd.DataFrame({'x_1': mXPca[:, 0], 'x_2': mXPca[:, 1], 'Label': vY})
dfDataLda = pd.DataFrame({'x_1': mXLda[:, 0], 'x_2': mXLda[:, 1], 'Label': vY})


# %% Plot Results

# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
sns.set_theme(style = 'ticks', context = 'talk')
plt.style.use('dark_background')  # inverts colors to dark theme

figIdx += 1

hF, vHa = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 6))
vHa = vHa.flat
sns.scatterplot(data = dfDataPca, x = 'x_1', y = 'x_2', hue = 'Label',
                palette = 'tab10', ax = vHa[0])
vHa[0].set_title('PCA')
sns.scatterplot(data = dfDataLda, x = 'x_1', y = 'x_2', hue = 'Label', 
                palette = 'tab10', ax = vHa[1])
vHa[1].set_title('LDA')
for hA in vHa:
    hA.set_xlabel('$x_1$')
    hA.set_ylabel('$x_2$')

hF.savefig(f'Figure{figIdx:04d}.svg', transparent = True)


# %%
