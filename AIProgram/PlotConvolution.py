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
from typing import Callable, List, Tuple, Union

# %% Configuration

# %matplotlib inline

# warnings.filterwarnings('ignore')

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

BG_COLOR      = "#202020"
PANEL_COLOR   = "#2b2b2b"
TEXT_COLOR    = "#f8fafc"
MUTED_COLOR   = "#94a3b8"
SIGNAL_COLOR  = "#38bdf8"
FILTER_COLOR  = "#f59e0b"
ACTIVE_COLOR  = "#facc15"
OUTPUT_COLOR  = "#34d399"

# %% Courses Packages


# %% Auxiliary Functions

def smooth_step(tt):
	"""Cubic easing with zero velocity at both ends."""
	return tt * tt * (3.0 - 2.0 * tt)


def draw_cell(ax, xx, yy, value, color, alpha=1.0, edge_color=None, zorder=2):
	"""Draw one numbered cell centered at (xx, yy)."""
	edge_color = color if edge_color is None else edge_color
	cell = FancyBboxPatch(
		(xx - 0.34, yy - 0.29), 0.68, 0.58,
		boxstyle="round,pad=0.02,rounding_size=0.06",
		facecolor=color, edgecolor=edge_color, linewidth=2.0,
		alpha=alpha, zorder=zorder
	)
	ax.add_patch(cell)
	txt = ax.text(
		xx, yy, f"{value:g}", ha="center", va="center",
		color=BG_COLOR, fontsize=14, fontweight="bold", zorder=zorder + 1
	)
	txt.set_path_effects([pe.withStroke(linewidth=1.2, foreground=color)])


def style_axis(ax):
	ax.set_facecolor(PANEL_COLOR)
	ax.grid(False)
	for spine in ax.spines.values():
		spine.set_visible(False)
	ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# %% Parameters

# Data
vSignal = np.array([2, 1, 3, -1, 2, 0, 1], dtype=float)
vFilter = np.array([1, -1, 2], dtype=float)

# Model
vFilterFlipped = vFilter[::-1]
vConv = np.convolve(vSignal, vFilter, mode="full")

# Visualization
numFlipHoldFrames = 10
numFlipFrames     = 24
numAfterFlipFrames = 8
numFramesPerStep = 10
numMoveFrames    = 6
fps              = 12
outputWidth      = 1000


# %% Loading / Generating Data


# %% Analyze Data


# %% Plot Results

numSignal = vSignal.size
numFilter = vFilter.size
numOutput = vConv.size
firstFilterPos = -(numFilter - 1)
numIntroFrames = numFlipHoldFrames + numFlipFrames + numAfterFlipFrames
numFrames = numIntroFrames + numOutput * numFramesPerStep

figWidth = 12.0
figHeight = 7.0
outputDpi = outputWidth / figWidth
fig = plt.figure(figsize=(figWidth, figHeight), facecolor="none")
gs = fig.add_gridspec(3, 1, height_ratios=[1.70, 2.75, 1.70], hspace=0.22)
axFlip = fig.add_subplot(gs[0])
axSlide = fig.add_subplot(gs[1])
axOutput = fig.add_subplot(gs[2])
fig.subplots_adjust(left=0.055, right=0.975, top=0.91, bottom=0.07)
fig.suptitle(
	"1D Convolution", color=TEXT_COLOR, fontsize=24, fontweight="bold", y=0.965
)


def draw_flip_stage(progress):
	axFlip.clear()
	style_axis(axFlip)
	finalFilterRight = firstFilterPos + numOutput - 1 + numFilter - 1
	axFlip.set_xlim(firstFilterPos - 1.45, finalFilterRight + 0.60)
	axFlip.set_ylim(-0.60, 1.10)

	axFlip.text(firstFilterPos - 0.45, 0.75, "FLIP THE FILTER", color=MUTED_COLOR,
				fontsize=11, fontweight="bold")
	center = 0.5 * (numSignal - 1)
	left = center - 0.5 * (numFilter - 1)
	for jj, value in enumerate(vFilter):
		start_x = left + jj
		end_x = left + numFilter - 1 - jj
		cell_x = (1.0 - progress) * start_x + progress * end_x
		draw_cell(axFlip, cell_x, 0.0, value, FILTER_COLOR)

	source_label = "Filter"
	axFlip.text(firstFilterPos - 0.80, 0.0, source_label, ha="right", va="center",
				color=TEXT_COLOR, fontsize=12)


def draw_slide_stage(filter_pos, output_idx, settled):
	axSlide.clear()
	style_axis(axSlide)
	finalFilterRight = firstFilterPos + numOutput - 1 + numFilter - 1
	axSlide.set_xlim(firstFilterPos - 1.45, finalFilterRight + 0.60)
	axSlide.set_ylim(-0.60, 2.15)

	axSlide.text(firstFilterPos - 0.45, 1.80, "SLIDE AND MULTIPLY", color=MUTED_COLOR,
				 fontsize=11, fontweight="bold")
	axSlide.text(firstFilterPos - 0.80, 0.0, "Signal", ha="right", va="center",
				 color=TEXT_COLOR, fontsize=12)
	axSlide.text(firstFilterPos - 0.80, 1.05, "Filter", ha="right", va="center",
				 color=TEXT_COLOR, fontsize=12)

	overlap_indices = []
	if settled and output_idx is not None:
		overlap_indices = [
			jj for jj in range(numFilter)
			if 0 <= firstFilterPos + output_idx + jj < numSignal
		]

	for ii, value in enumerate(vSignal):
		is_active = settled and any(
			np.isclose(filter_pos + jj, ii) for jj in overlap_indices
		)
		draw_cell(
			axSlide, ii, 0.0, value,
			ACTIVE_COLOR if is_active else SIGNAL_COLOR,
			edge_color=ACTIVE_COLOR if is_active else SIGNAL_COLOR
		)

	for jj, value in enumerate(vFilterFlipped):
		is_active = jj in overlap_indices
		draw_cell(
			axSlide, filter_pos + jj, 1.05, value,
			ACTIVE_COLOR if is_active else FILTER_COLOR,
			alpha=1.0 if settled else 0.82,
			edge_color=ACTIVE_COLOR if is_active else FILTER_COLOR,
			zorder=4
		)

def draw_output_stage(numCompleted, active_idx):
	axOutput.clear()
	style_axis(axOutput)
	finalFilterRight = firstFilterPos + numOutput - 1 + numFilter - 1
	axOutput.set_xlim(firstFilterPos - 1.45, finalFilterRight + 0.60)
	axOutput.set_ylim(-0.60, 1.10)
	axOutput.text(firstFilterPos - 0.45, 0.75, "CONVOLUTION OUTPUT",
				  color=MUTED_COLOR, fontsize=11, fontweight="bold")
	axOutput.text(firstFilterPos - 0.80, 0.0, "Output", ha="right", va="center",
				  color=TEXT_COLOR, fontsize=12)

	if numCompleted > 0:
		for ii in range(numCompleted):
			edge_color = ACTIVE_COLOR if ii == active_idx else OUTPUT_COLOR
			draw_cell(axOutput, ii, 0.0, vConv[ii], OUTPUT_COLOR,
					  edge_color=edge_color, zorder=3)


def update(frame_idx):
	if frame_idx < numFlipHoldFrames:
		flip_progress = 0.0
	elif frame_idx < numFlipHoldFrames + numFlipFrames:
		tt = (frame_idx - numFlipHoldFrames) / (numFlipFrames - 1)
		flip_progress = smooth_step(tt)
	else:
		flip_progress = 1.0

	draw_flip_stage(flip_progress)

	if frame_idx < numIntroFrames:
		draw_slide_stage(firstFilterPos, None, False)
		draw_output_stage(0, None)
		return

	slide_frame = frame_idx - numIntroFrames
	output_idx = min(slide_frame // numFramesPerStep, numOutput - 1)
	local_frame = slide_frame % numFramesPerStep
	previous_pos = firstFilterPos + max(output_idx - 1, 0)
	target_pos = firstFilterPos + output_idx

	if output_idx == 0 or local_frame >= numMoveFrames:
		filter_pos = target_pos
		settled = True
	else:
		tt = local_frame / (numMoveFrames - 1)
		filter_pos = previous_pos + smooth_step(tt)
		settled = local_frame == numMoveFrames - 1

	num_completed = output_idx + 1 if settled else output_idx
	draw_slide_stage(filter_pos, output_idx, settled)
	draw_output_stage(num_completed, output_idx if settled else output_idx - 1)


anim = FuncAnimation(fig, update, frames=numFrames, interval=1000 / fps, repeat=True)
outputPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Convolution1D.png")
anim.save(
	outputPath,
	writer=PillowWriter(fps=fps),
	dpi=outputDpi
)
print(f"Saved animation to: {outputPath}")
if plt.get_backend().lower() != "agg":
	plt.show()

# %%
