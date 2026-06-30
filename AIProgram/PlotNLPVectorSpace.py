# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot the Vector Space of NLP Models / Text Embeddings
# Visualization of simple arithmetic in 3D Space.
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
# | 0.1.000 | 29/06/2026 | Royi Avital | First version                                                      |
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

def draw_vec(ax, p, color, lw=2.2, alpha=1.0):
    """Draw vector from origin to p."""
    ax.quiver(
        0, 0, 0,
        p[0], p[1], p[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.10,
        alpha=alpha
    )

def draw_arrow_between(ax, p0, p1, color, lw=2.8):
    """Draw arrow from p0 to p1."""
    d = p1 - p0
    ax.quiver(
        p0[0], p0[1], p0[2],
        d[0], d[1], d[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.22
    )

def glow_text(ax, x, y, z, s, color, size=12):
    txt = ax.text(x, y, z, s, color=color, fontsize=size)
    txt.set_path_effects([
        pe.withStroke(linewidth=3, foreground="black")
    ])
    return txt

# %% Parameters

# Data
vManEmbedding   = np.array([ 2.4, 1.4, 1.45])
vWomanEmbedding = np.array([ 1.75, 0.95, 0.65])

vUncleEmbedding = np.array([-1.8, 1.25, 1.35])
vAuntEmbedding  = np.array([-2.45, 0.80, 0.55])

uncleColor = "#77bde0"
auntColor  = "#ff6f7f"
manColor   = "#e7eefb"
womanColor = "#ff7f8c"
diffColor  = "#f5df4d"

# Model
polyDeg    = 15
paramC     = 1e9
kernelType = 'linear'


# Visualization
numGridPts = 500


# %% Loading / Generating Data




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

hF = plt.figure(figsize = (8, 8), facecolor = 'black')
hA = hF.add_subplot(projection = "3d")
hA.set_facecolor('black')

# View similar to the reference image
hA.view_init(elev = 18, azim = -67)
hA.set_box_aspect((1.5, 1.2, 0.9))

# Limits
hA.set_xlim(-3.2, 3.2)
hA.set_ylim(-1.4, 2.7)
hA.set_zlim(-0.1, 2.5)

# Hide default panes and ticks
hA.set_xticks([])
hA.set_yticks([])
hA.set_zticks([])
hA.set_xlabel('')
hA.set_ylabel('')
hA.set_zlabel('')

for axis in [hA.xaxis, hA.yaxis, hA.zaxis]:
    axis.pane.set_facecolor((0, 0, 0, 0))
    axis.pane.set_edgecolor((0, 0, 0, 0))
    axis.line.set_color((0, 0, 0, 0))

tuGridColor = (0.65, 0.65, 0.65, 0.45)
vXVals = np.linspace(-3.2, 3.2, 13)
vYVals = np.linspace(-1.4, 2.7, 11)

for x in vXVals:
    hA.plot([x, x], [vYVals[0], vYVals[-1]], [0, 0],
            color = tuGridColor, lw = 0.7)

for y in vYVals:
    hA.plot([vXVals[0], vXVals[-1]], [y, y], [0, 0],
            color = tuGridColor, lw = 0.7)

# Main axes
axisColor = (0.9, 0.9, 0.9, 0.9)
hA.plot([-3.3, 3.3], [0, 0], [0, 0], color = axisColor, lw = 1.4)
hA.plot([0, 0], [-1.5, 2.8], [0, 0], color = axisColor, lw = 1.4)
hA.plot([0, 0], [0, 0], [0, 2.6], color = axisColor, lw = 1.4)

# Small tick marks on vertical axis
for z in np.linspace(0.35, 2.35, 6):
    hA.plot([-0.05, 0.05], [0, 0], [z, z],
            color = axisColor, lw = 1.0)


draw_vec(hA, vUncleEmbedding, uncleColor)
draw_vec(hA, vAuntEmbedding, auntColor)
draw_vec(hA, vManEmbedding, manColor)
draw_vec(hA, vWomanEmbedding, womanColor)

# Difference arrows:
# E(aunt) - E(uncle) and E(woman) - E(man)
draw_arrow_between(hA, vUncleEmbedding, vAuntEmbedding, diffColor)
draw_arrow_between(hA, vManEmbedding, vWomanEmbedding, diffColor)

# -----------------------------
# Labels
# -----------------------------
glow_text(hA, vUncleEmbedding[0] - 0.45, vUncleEmbedding[1] + 0.05, vUncleEmbedding[2] + 0.12,
          r"$E(\mathrm{uncle})$", uncleColor, 11)

glow_text(hA, vAuntEmbedding[0] - 0.25, vAuntEmbedding[1] - 1.00, vAuntEmbedding[2] + 0.08,
          r"$E(\mathrm{aunt})$", auntColor, 12)

glow_text(hA, vManEmbedding[0] + 0.05, vManEmbedding[1] + 0.05, vManEmbedding[2] + 0.10,
          r"$E(\mathrm{man})$", manColor, 12)

glow_text(hA, vWomanEmbedding[0] + 0.35, vWomanEmbedding[1] - 1.20, vWomanEmbedding[2] + 0.08,
          r"$E(\mathrm{woman})$", womanColor, 12)

# Title and equation
# hF.text(
#     0.5, 0.93,
#     "Visualizing Words\nin Vector Space",
#     ha = "center", va = "center",
#     color = "white",
#     fontsize = 26
# )

# hF.text(
#     0.5, 0.78,
#     r"$\mathbf{E}(\mathrm{aunt}) - \mathbf{E}(\mathrm{uncle})"
#     r"\ \approx\ "
#     r"\mathbf{E}(\mathrm{woman}) - \mathbf{E}(\mathrm{man})$",
#     ha = "center", va = "center",
#     color = "#f08a8a",
#     fontsize = 17
# )

hF.tight_layout(pad = 0)
hF.show()

hF.savefig('TMP.svg', transparent = True)


# %%
