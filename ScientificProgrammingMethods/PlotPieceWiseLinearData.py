# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Dynamic Programming - Piece Wise Linear Data
# Generates the piece wise linear data and plots it.
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
# | 0.1.000 | 01/05/2025 | Royi Avital | First version                                                      |
# |         |            |             |                                                                    |

# %% Packages

# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning

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

# Color Palettes
lMatPltLibclr   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] #<! Matplotlib default color palette
lFlexokiClr     = ['#D14D41', '#DA702C', '#D0A215', '#879A39', '#3AA99F', '#4385BE', '#8B7EC8', '#CE5D97'] #<! Flexoki (Obsidian) Main
lFlexokiSatClr  = ['#AF3029', '#BC5215', '#BC5215', '#66800B', '#24837B', '#205EA6', '#5E409D', '#A02F6F'] #<! Flexoki (Obsidian) Saturated
lFlexokiGrayClr = ['#100F0F', '#1C1B1B', '#282726', '#343331', '#403E3C', '#55524E', '#878580', '#CECDC3'] #<! Flexoki (Obsidian) Grayscale

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants


# %% Courses Packages


# %% Auxiliary Functions

def GenPieceWiseLinearData( numSamples: int, numSegments: int, /, *, tuXRange: Tuple[float, float] = (0, 1), σ: float = 0.0, seedNum: int = 123 ) -> np.ndarray:
    """
    Generates samples (x, y) from a piecewise linear function.

    Parameters:
    -----------
    numSamples : int
        Total number of (x, y) samples to generate.
    numSegments : int
        Number of linear segments in the piecewise model.
    tuXRange : tuple
        Range of x values, e.g. (0, 1).
    σ : float
        Standard deviation of Gaussian noise added to y.

    Returns:
    --------
    x : ndarray
        Array of shape (numSamples,) of x-values.
    y : ndarray
        Array of shape (numSamples,) of y-values from the piecewise model.
    """

    oRng = np.random.default_rng(seed = seedNum)

    # Generate evenly spaced x-values
    vX = np.linspace(tuXRange[0], tuXRange[1], numSamples)

    # Create breakpoints
    vB = np.linspace(tuXRange[0], tuXRange[1], numSegments + 1)

    # Assign random slopes and intercepts
    vS = oRng.uniform(-5, 5, size = numSegments)
    vI = np.zeros(numSegments)

    # Compute intercepts so the function is continuous
    for ii in range(1, numSegments):
        xPrev  = vB[ii]
        vI[ii] = (vS[ii - 1] * xPrev + vI[ii - 1]) - vS[ii] * xPrev

    # Assign y values based on the segment each x belongs to
    vY = np.zeros_like(vX)
    for ii in range(numSegments):
        vM     = (vX >= vB[ii]) & (vX <= vB[ii + 1])
        vY[vM] = vS[ii] * vX[vM] + vI[ii]

    # 6. Add  noise
    vY += oRng.normal(scale = σ, size = numSamples)

    return vX, vY

# def GenPieceWiseLinearData( numSamples: int, numSegments: int, /, *, tuXRange: Tuple[float, float] = (0, 1), σ: float = 0.0, seedNum: int = 123 ) -> np.ndarray:
#     """
#     Generates samples (x, y) from a piecewise linear function.

#     Parameters:
#     -----------
#     numSamples : int
#         Total number of (x, y) samples to generate.
#     numSegments : int
#         Number of linear segments in the piecewise model.
#     tuXRange : tuple
#         Range of x values, e.g. (0, 1).
#     σ : float
#         Standard deviation of Gaussian noise added to y.

#     Returns:
#     --------
#     x : ndarray
#         Array of shape (numSamples,) of x-values.
#     y : ndarray
#         Array of shape (numSamples,) of y-values from the piecewise model.
#     """

#     oRng = np.random.default_rng(seed = seedNum)

#     # Generate evenly spaced x-values
#     vX = oRng.uniform(tuXRange[0], tuXRange[1], numSamples)

#     # Assign random slopes and intercepts
#     mL = oRng.uniform(-5, 5, size = (numSegments, 2)) #<! Line Parameters: (Slope, Intercept)
#     mL[:, 0] = np.sort(mL[:, 0]) #<! Sort slopes to ensure increasing order
#     mL[:, 1] = np.sort(mL[:, 1])[::-1] #<! Sort intercepts to ensure decreasing order
#     # mL = oRng.normal(0, 2, size = (numSegments, 2)) #<! Line Parameters: (Slope, Intercept)

#     # Calculate values
#     mY = mL @ np.vstack((vX, np.ones(numSamples))) #<! Matrix multiplication to get y values
#     vY = np.max(mY, axis = 0) #<! Get the maximum y value for each x

#     # Add  noise
#     vY += oRng.normal(scale = σ, size = numSamples)

#     return vX, vY



# %% Parameters

# Data
numSamples  = 45
numSegments = 5
tuXRange    = (0, 2)
σ           = 0.1
seedNum     = 110

# Model

# Visualization


# %% Loading / Generating Data

vX, vY = GenPieceWiseLinearData(numSamples, numSegments, tuXRange = tuXRange, σ = 0.0, seedNum = seedNum)


# %% Analyze Data

mX  = np.c_[vX, vY]
dfX = pd.DataFrame(mX, columns = ['x', 'y'])
# dfX.to_csv('PieceWiseLinearData.csv', index = False)


# %% Plot Results

# sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
# sns.color_palette('tab10')
sns.set_theme(style = 'ticks', context = 'talk')
plt.style.use('dark_background')  # inverts colors to dark theme

hF, hA = plt.subplots(figsize = (10, 6))
sns.scatterplot(x = vX, y = vY,
                ax = hA, color = lFlexokiSatClr[0], edgecolor = 'white')
hA.set_title('Data Samples')

hA.set_xlabel('$x$')
hA.set_ylabel('$y$');

# hF.savefig('TMP.svg', transparent = True)


# %%
