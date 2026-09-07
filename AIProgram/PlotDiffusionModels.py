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
exportFig  = True
exportDir  = 'Figures'

# %% Load / Generate Data

# Ensure export directory exists

os.makedirs(exportDir, exist_ok = True)


# %% Noise Evolution

tI = iio.imread(owlImgUrl) / 255.0

for t in range(numGridPtsT):
    tN = np.random.normal(0, lNoiseStd[t], tI.shape)
    tI = np.clip(tI + tN, 0, 1)
    plt.imshow(tI)
    plt.title(f'Noise Evolution - Step {t + 1}')
    plt.axis('off')
    plt.show()
    if exportFig:
        fileName = f'NoiseEvolution_Step{t + 1}.png'
        filePath = os.path.join(exportDir, fileName)
        iio.imwrite(filePath, np.round(tI * 255).astype(np.uint8))

    if t == 0 and exportFig:
        fileName = f'Noise.png'
        filePath = os.path.join(exportDir, fileName)
        iio.imwrite(filePath, np.round(tN * 255).astype(np.uint8))


# %% Visualize 2D Mixture

numClusters   = 6
numSamples    = 3000
clusterRadius = 4.0
clusterStd    = 0.25
numDiffSteps  = 8
axisLimit     = 6.0

oRng = np.random.default_rng(seedNum)
vAngles = np.linspace(0, 2 * np.pi, numClusters, endpoint = False)
mCenters = clusterRadius * np.column_stack((np.cos(vAngles), np.sin(vAngles)))
vLabels = np.arange(numSamples) % numClusters
mSamples = mCenters[vLabels] + clusterStd * oRng.standard_normal((numSamples, 2))
mColors = plt.get_cmap('tab10')(np.arange(numClusters))
ɑBar = 1.0 #<! Initial value 
ɑBarEnd = 0.05 #<! Final value for ɑBar

β = 1 - (ɑBarEnd / ɑBar) ** (1 / numDiffSteps) #<! Adaptive based on number of iterations and the desired final value of ɑBar

with plt.style.context('dark_background'):
    for stepIdx in range(numDiffSteps + 1):
        if stepIdx > 0:
            mNoise = oRng.standard_normal(mSamples.shape)
            ɑ = 1 - β
            mSamples = np.sqrt(ɑ) * mSamples + np.sqrt(β) * mNoise
            ɑBar *= ɑ #<! Accumulation of the product of ɑ's over time, which is used to compute the effective noise level at each step

        hF, hA = plt.subplots(figsize = (8, 8))
        hA.scatter(mSamples[:, 0], mSamples[:, 1], c = mColors[vLabels], s = 8, alpha = 0.55, edgecolors = 'none')
        hA.set_xlim(-axisLimit, axisLimit)
        hA.set_ylim(-axisLimit, axisLimit)
        hA.set_aspect('equal', adjustable = 'box')
        hA.set_xticks(np.arange(-axisLimit, axisLimit + 1, 2))
        hA.set_yticks(np.arange(-axisLimit, axisLimit + 1, 2))
        hA.set_xlabel('$x_1$', fontsize = 14)
        hA.set_ylabel('$x_2$', fontsize = 14)
        hA.grid(True, color = 'white', alpha = 0.12)
        hA.set_axisbelow(True)
        hA.text(0.03, 0.97, rf'$\sqrt{{\bar\alpha_t}}$ = {np.sqrt(ɑBar):.3f}',
            transform = hA.transAxes, ha = 'left', va = 'top', fontsize = 14)
        hF.tight_layout(pad = 0.2)

        if exportFig:
            fileName = f'DiffusionMixture_Step{stepIdx:02d}.png'
            filePath = os.path.join(exportDir, fileName)
            hF.savefig(filePath, dpi = 150, facecolor = 'none', transparent = True, bbox_inches = 'tight', pad_inches = 0.02)

        plt.show()
        plt.close(hF)

# %% Plot the Flow Matching Point of View

numFlowSteps = 20
numFlowSamples = 3000
flowAlphaBarEnd = 1e-4
flowAxisLimit = 7.0
flowGridSize = 220

mFlowMeans = np.array([[-3.5, -2.0], [-2.5, 2.8], [0.5, 0.0], [3.0, 2.5], [3.5, -2.5]])
tFlowCov = np.array([
    [[0.65, 0.40], [0.40, 0.45]],
    [[0.25, -0.15], [-0.15, 0.80]],
    [[0.70, -0.35], [-0.35, 0.35]],
    [[0.50, 0.25], [0.25, 0.50]],
    [[0.30, 0.00], [0.00, 0.65]],
])
vFlowWeights = np.array([0.25, 0.15, 0.20, 0.25, 0.15])
oFlowRng = np.random.default_rng(seedNum)
vFlowLabels = oFlowRng.choice(len(vFlowWeights), size = numFlowSamples, p = vFlowWeights)
mFlowSamples = np.empty((numFlowSamples, 2))
vTrackIdx = np.empty(len(vFlowWeights), dtype = int)
for clusterIdx in range(len(vFlowWeights)):
    vClusterIdx = np.flatnonzero(vFlowLabels == clusterIdx)
    mFlowSamples[vClusterIdx] = oFlowRng.multivariate_normal(mFlowMeans[clusterIdx], tFlowCov[clusterIdx], size = len(vClusterIdx))
    vTrackIdx[clusterIdx] = oFlowRng.choice(vClusterIdx)

tFlowTracks = np.empty((numFlowSteps + 1, len(vTrackIdx), 2))
mTrackColors = plt.get_cmap('tab10')(np.arange(len(vTrackIdx)))
flowAlpha = flowAlphaBarEnd ** (1 / numFlowSteps)
vFlowAlphaBar = flowAlpha ** np.arange(numFlowSteps + 1)
vFlowGrid = np.linspace(-flowAxisLimit, flowAxisLimit, flowGridSize)
mFlowXX, mFlowYY = np.meshgrid(vFlowGrid, vFlowGrid)
tFlowGrid = np.stack((mFlowXX, mFlowYY), axis = -1)
mNormalDensity = sp.stats.multivariate_normal.pdf(tFlowGrid, mean = np.zeros(2), cov = np.eye(2))
vDensityLevels = np.array([0.003, 0.01, 0.03, 0.06, 0.10, 0.15])

with plt.style.context('dark_background'):
    for stepIdx, alphaBar in enumerate(vFlowAlphaBar):
        if stepIdx > 0:
            mFlowNoise = oFlowRng.standard_normal(mFlowSamples.shape)
            mFlowSamples = np.sqrt(flowAlpha) * mFlowSamples + np.sqrt(1 - flowAlpha) * mFlowNoise
        tFlowTracks[stepIdx] = mFlowSamples[vTrackIdx]

        mFlowDensity = np.zeros_like(mFlowXX)
        for clusterIdx, weight in enumerate(vFlowWeights):
            vMean = np.sqrt(alphaBar) * mFlowMeans[clusterIdx]
            mCov = alphaBar * tFlowCov[clusterIdx] + (1 - alphaBar) * np.eye(2)
            mFlowDensity += weight * sp.stats.multivariate_normal.pdf(tFlowGrid, mean = vMean, cov = mCov)

        hF, hA = plt.subplots(figsize = (8, 8))
        hA.scatter(mFlowSamples[:, 0], mFlowSamples[:, 1], color = 'white', s = 4, alpha = 0.12, edgecolors = 'none')
        hA.contour(mFlowXX, mFlowYY, mNormalDensity, levels = vDensityLevels, colors = '#a0a0a0', linestyles = 'dashed', linewidths = 0.8, alpha = 0.7)
        hA.contour(mFlowXX, mFlowYY, mFlowDensity, levels = vDensityLevels, colors = 'cyan', linewidths = 1.0)
        hA.plot([], [], color = 'cyan', lw = 1.0, label = r'$p_t$: current density')
        hA.plot([], [], color = '#a0a0a0', lw = 0.8, ls = '--', label = r'$p_{\mathrm{init}}$: standard normal')

        for trackIdx, color in enumerate(mTrackColors):
            mTrail = tFlowTracks[:stepIdx + 1, trackIdx]
            hA.plot(mTrail[:, 0], mTrail[:, 1], color = color, lw = 1.5, alpha = 0.8)
            hA.scatter(*mTrail[0], s = 70, facecolors = 'none', edgecolors = [color], linewidths = 1.5, zorder = 4)
            hA.scatter(*mTrail[-1], s = 85, color = color, edgecolors = 'white', linewidths = 0.7, zorder = 5)
            hA.annotate(str(trackIdx + 1), mTrail[-1], xytext = (6, 6), textcoords = 'offset points', color = color, fontsize = 11, clip_on = True)

        hA.set(xlim = (-flowAxisLimit, flowAxisLimit), ylim = (-flowAxisLimit, flowAxisLimit), aspect = 'equal')
        hA.set_xticks(np.arange(-6, 7, 2))
        hA.set_yticks(np.arange(-6, 7, 2))
        hA.set_xlabel('$x_1$', fontsize = 14)
        hA.set_ylabel('$x_2$', fontsize = 14)
        hA.grid(True, color = 'white', alpha = 0.12)
        hA.set_axisbelow(True)
        hA.text(0.03, 0.97, rf'$t = {stepIdx:02d}$, $\sqrt{{\bar\alpha_t}} = {np.sqrt(alphaBar):.3f}$', transform = hA.transAxes, ha = 'left', va = 'top', fontsize = 14)
        hA.legend(loc = 'lower left', frameon = False, fontsize = 10)
        hF.tight_layout(pad = 0.2)

        if exportFig:
            filePath = os.path.join(exportDir, f'DiffusionTrackedSamples_Step{stepIdx:02d}.png')
            hF.savefig(filePath, dpi = 150, facecolor = 'none', transparent = True, bbox_inches = 'tight', pad_inches = 0.02)

        plt.show()
        plt.close(hF)

# Plot Endpoint Contours
with plt.style.context('dark_background'):
    for endpointName, alphaBar in [('Start', 1.0), ('End', flowAlphaBarEnd)]:
        mEndpointDensity = np.zeros_like(mFlowXX)
        for clusterIdx, weight in enumerate(vFlowWeights):
            vMean = np.sqrt(alphaBar) * mFlowMeans[clusterIdx]
            mCov = alphaBar * tFlowCov[clusterIdx] + (1 - alphaBar) * np.eye(2)
            mEndpointDensity += weight * sp.stats.multivariate_normal.pdf(tFlowGrid, mean = vMean, cov = mCov)

        hF, hA = plt.subplots(figsize = (8, 8))
        hA.contour(mFlowXX, mFlowYY, mEndpointDensity, levels = vDensityLevels, colors = 'cyan', linewidths = 1.0)
        hA.set(xlim = (-flowAxisLimit, flowAxisLimit), ylim = (-flowAxisLimit, flowAxisLimit), aspect = 'equal')
        hA.set_xticks(np.arange(-6, 7, 2))
        hA.set_yticks(np.arange(-6, 7, 2))
        hA.set_xlabel('$x_1$', fontsize = 14)
        hA.set_ylabel('$x_2$', fontsize = 14)
        hA.grid(True, color = 'white', alpha = 0.12)
        hA.set_axisbelow(True)
        hF.tight_layout(pad = 0.2)

        if exportFig:
            filePath = os.path.join(exportDir, f'DiffusionContour_{endpointName}.png')
            hF.savefig(filePath, dpi = 150, facecolor = 'none', transparent = True, bbox_inches = 'tight', pad_inches = 0.02)

        plt.show()
        plt.close(hF)


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
