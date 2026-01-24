# Transposed Convolution 2D Layer
# Visualizations for the Transposed Convolution 2D Layer (`ConvTranspose2d`) Slides
# References:
#   1.  A
# Remarks:
#   1.  Use in Julia as following:
#       -   Move to folder using `cd(raw"<PathToFolder>");`.
#       -   Activate the environment using `] activate .`.
#       -   Instantiate the environment using `] instantiate`.
#   2.  A
# TODO:
# 	1.  AA.
# Release Notes Royi Avital RoyiAvital@yahoo.com
# - 1.0.000     24/01/2021  Royi Avital
#   *   First release.

## Packages

# Internal
using LinearAlgebra;
using Printf;
using Random;
# External
using BenchmarkTools;
using Images;
using Interpolations;
using Luxor;
# using PlotlyJS;            #<! Use `add Kaleido_jll@v0.1;` (See https://github.com/JuliaPlots/PlotlyJS.jl/issues/479)
using StableRNGs;


## Constants & Configuration
RNG_SEED = 1234;


## Settings

figureIdx = 0;

exportFigures = true;

oRng = StableRNG(1234);


## Functions


## Parameters

# Canvas
tuCanvasSize = (800, 600); #<! (width, height)

# Grid
cellSize = 40;

# Input
tuImageSize  = (2, 2);
tuKernelSize = (3, 3);

# Colors
gridClr = Luxor.Colors.RGBA(1.0, 1.0, 1.0, 1.0);
txtClr  = Luxor.Colors.RGBA(1.0, 1.0, 1.0, 1.0);


## Drawing

figureIdx += 1;

canA = Drawing(tuCanvasSize[1], tuCanvasSize[2], :rec);


finish();


