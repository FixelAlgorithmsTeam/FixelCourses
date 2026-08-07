# %% [markdown]
#
# [![Fixel Algorithms](https://fixelalgorithms.co/images/CCExt.png)](https://fixelalgorithms.gitlab.io)
# 
# # Plot Figures - Plot Animation of 1D Convolution Operation
# Visualization of 1D Convolution.
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
# | 0.1.000 | 07/08/2026 | Royi Avital | First version                                                      |
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
from matplotlib import patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import seaborn as sns

# Typing
from typing import Callable, List, Optional, Tuple, Union

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

bgColor     = '#202020'
panelColor  = '#2b2b2b'
textColor   = '#f8fafc'
mutedColor  = '#94a3b8'
signalColor = '#38bdf8'
filterColor = '#f59e0b'
activeColor = '#facc15'
outputColor = '#34d399'

# %% Courses Packages


# %% Auxiliary Functions

def SmoothStep(tt: float) -> float:
	'''Cubic easing with zero velocity at both ends.'''
	return tt * tt * (3.0 - 2.0 * tt)


def DrawCell(ax: plt.Axes, xx: float, yy: float, value: float, color: str, alpha: float = 1.0,
			 edgeColor: Optional[str] = None, zOrder: int = 2) -> None:
	'''Draw one numbered cell centered at (xx, yy).'''
	edgeColor = color if edgeColor is None else edgeColor
	hCell = FancyBboxPatch(
		(xx - 0.34, yy - 0.29), 0.68, 0.58,
		boxstyle = 'round,pad=0.02,rounding_size=0.06',
		facecolor = color, edgecolor = edgeColor, linewidth = 2.0,
		alpha = alpha, zorder = zOrder
	)
	ax.add_patch(hCell)
	hText = ax.text(
		xx, yy, f'{value:g}', ha = 'center', va = 'center',
		color = bgColor, fontsize = 14, fontweight = 'bold', zorder = zOrder + 1
	)
	hText.set_path_effects([pe.withStroke(linewidth = 1.2, foreground = color)])


def StyleAxis(ax: plt.Axes) -> None:
	ax.set_facecolor(panelColor)
	ax.grid(False)
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.tick_params(left = False, bottom = False, labelleft = False, labelbottom = False)


# %% Parameters

# Data
fileName = 'Convolution1D.png'
vSignal  = np.array([2, 1, 3, -1, 2, 0, 1])
vFilter  = np.array([1, -1, 2])

# Model
vFlipped = vFilter[::-1]
vConv    = np.convolve(vSignal, vFilter, mode = 'full')

# Visualization
holdFrames  = 10
flipFrames  = 24
afterFrames = 8
stepFrames  = 10
moveFrames  = 6
frameRate   = 12
outputWidth = 1000


# %% Loading / Generating Data


# %% Analyze Data


# %% Plot Results

signalSize  = vSignal.size
filterSize  = vFilter.size
outputSize  = vConv.size
firstPos    = -(filterSize - 1)
introFrames = holdFrames + flipFrames + afterFrames
numFrames   = introFrames + outputSize * stepFrames

figWidth = 12.0
figHeight = 7.0
outputDpi = outputWidth / figWidth

hF        = plt.figure(figsize = (figWidth, figHeight), facecolor = 'none')
hGridSpec = hF.add_gridspec(3, 1, height_ratios = [1.70, 2.75, 1.70], hspace = 0.22)
hAFlip    = hF.add_subplot(hGridSpec[0])
hASlide   = hF.add_subplot(hGridSpec[1])
hAOutput  = hF.add_subplot(hGridSpec[2])

hF.subplots_adjust(left = 0.055, right = 0.975, top = 0.91, bottom = 0.07)
hF.suptitle('1D Convolution', color = textColor, fontsize = 24, fontweight = 'bold', y = 0.965)

def DrawFlipStage(progress: float) -> None:
	hAFlip.clear()
	StyleAxis(hAFlip)
	finalRight = firstPos + outputSize - 1 + filterSize - 1
	hAFlip.set_xlim(firstPos - 1.45, finalRight + 0.60)
	hAFlip.set_ylim(-0.60, 1.10)

	hAFlip.text(firstPos - 0.45, 0.75, 'FLIP THE FILTER', color = mutedColor, fontsize = 11, fontweight = 'bold')
	centerPos = 0.5 * (signalSize - 1)
	leftPos = centerPos - 0.5 * (filterSize - 1)
	for jj, value in enumerate(vFilter):
		startPos = leftPos + jj
		endPos = leftPos + filterSize - 1 - jj
		cellPos = (1.0 - progress) * startPos + progress * endPos
		DrawCell(hAFlip, cellPos, 0.0, value, filterColor)

	textLabel = 'Filter'
	hAFlip.text(firstPos - 0.80, 0.0, textLabel, ha = 'right', va = 'center', color = textColor, fontsize = 12)


def DrawSlideStage(filterPos: float, outputIdx: Optional[int], isSettled: bool) -> None:
	hASlide.clear()
	StyleAxis(hASlide)
	finalRight = firstPos + outputSize - 1 + filterSize - 1
	hASlide.set_xlim(firstPos - 1.45, finalRight + 0.60)
	hASlide.set_ylim(-0.60, 2.15)

	hASlide.text(firstPos - 0.45, 1.80, 'SLIDE AND MULTIPLY', color = mutedColor, fontsize = 11, fontweight = 'bold')
	hASlide.text(firstPos - 0.80, 0.0, 'Signal', ha = 'right', va = 'center', color = textColor, fontsize = 12)
	hASlide.text(firstPos - 0.80, 1.05, 'Filter', ha = 'right', va = 'center', color = textColor, fontsize = 12)

	overlapIdx = []
	if isSettled and outputIdx is not None:
		overlapIdx = [
			jj for jj in range(filterSize)
			if 0 <= firstPos + outputIdx + jj < signalSize
		]

	for ii, value in enumerate(vSignal):
		isActive = isSettled and any(
			np.isclose(filterPos + jj, ii) for jj in overlapIdx
		)
		DrawCell(
			hASlide, ii, 0.0, value,
			activeColor if isActive else signalColor,
			edgeColor = activeColor if isActive else signalColor
		)

	for jj, value in enumerate(vFlipped):
		isActive = jj in overlapIdx
		DrawCell(
			hASlide, filterPos + jj, 1.05, value,
			activeColor if isActive else filterColor,
			alpha = 1.0 if isSettled else 0.82,
			edgeColor = activeColor if isActive else filterColor,
			zOrder = 4
		)

def DrawOutputStage(numCompleted: int, activeIdx: Optional[int]) -> None:
	hAOutput.clear()
	StyleAxis(hAOutput)
	finalRight = firstPos + outputSize - 1 + filterSize - 1
	hAOutput.set_xlim(firstPos - 1.45, finalRight + 0.60)
	hAOutput.set_ylim(-0.60, 1.10)
	hAOutput.text(firstPos - 0.45, 0.75, 'CONVOLUTION OUTPUT', color = mutedColor, fontsize = 11, fontweight = 'bold')
	hAOutput.text(firstPos - 0.80, 0.0, 'Output', ha = 'right', va = 'center', color = textColor, fontsize = 12)

	if numCompleted > 0:
		for ii in range(numCompleted):
			edgeColor = activeColor if ii == activeIdx else outputColor
			DrawCell(hAOutput, ii, 0.0, vConv[ii], outputColor,
					 edgeColor = edgeColor, zOrder = 3)


def Update(frameIdx: int) -> None:
	if frameIdx < holdFrames:
		flipProgress = 0.0
	elif frameIdx < holdFrames + flipFrames:
		tt = (frameIdx - holdFrames) / (flipFrames - 1)
		flipProgress = SmoothStep(tt)
	else:
		flipProgress = 1.0

	DrawFlipStage(flipProgress)

	if frameIdx < introFrames:
		DrawSlideStage(firstPos, None, False)
		DrawOutputStage(0, None)
		return

	slideFrame  = frameIdx - introFrames
	outputIdx   = min(slideFrame // stepFrames, outputSize - 1)
	localFrame  = slideFrame % stepFrames
	previousPos = firstPos + max(outputIdx - 1, 0)
	targetPos   = firstPos + outputIdx

	if outputIdx == 0 or localFrame >= moveFrames:
		filterPos = targetPos
		isSettled = True
	else:
		tt = localFrame / (moveFrames - 1)
		filterPos = previousPos + SmoothStep(tt)
		isSettled = localFrame == moveFrames - 1

	numCompleted = outputIdx + 1 if isSettled else outputIdx
	DrawSlideStage(filterPos, outputIdx, isSettled)
	DrawOutputStage(numCompleted, outputIdx if isSettled else outputIdx - 1)


hAnim = FuncAnimation(hF, Update, frames = numFrames, interval = 1000 / frameRate, repeat = True)
outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), fileName)
hAnim.save(outputPath,writer = PillowWriter(fps = frameRate), dpi = outputDpi)

print(f'Saved animation to: {outputPath}')
if plt.get_backend().lower() != 'agg':
	plt.show()

# %%
