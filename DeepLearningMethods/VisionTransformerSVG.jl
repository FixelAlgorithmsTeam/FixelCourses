# Vision Transformer
# Visualizations for the Vision Transformer (ViT) Slides
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
# - 1.0.000     20/12/2025  Royi Avital
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

# Image
downScaleFctr = 50;
upScaleFctr   = 24;
refPxIdx      = 185; #<! (17, 8)


## Drawing

mG = load("ColourfulParrot.png"); #<! Result of Gemini
mI = mG[(end - 1200 + 1):end, :];
# mI = load("ColourfulParrot1200800.png"); #<! From Paint.Net which altered color profile
tuImgSize = size(mI); #<! (numRows, numCols)
tuOutSize = (tuImgSize[1] ÷ 50, tuImgSize[2] ÷ 50);


mS = imresize(mI, tuOutSize);
mU = imresize(mS, (tuOutSize[1] * upScaleFctr, tuOutSize[2] * upScaleFctr), method = Constant());
mU = Luxor.Colors.ARGB32.(mU);

tuSizeU = size(mU);

canA = Drawing(tuCanvasSize[1], tuCanvasSize[2], :rec);
itrTile = Tiler(tuSizeU[2], tuSizeU[1], tuOutSize[1], tuOutSize[2], margin = 0);
for (pos, n) in itrTile
    sethue(mS'[n]);
    pos += Point(250, 0) + Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2);
    box(pos, upScaleFctr, upScaleFctr, :fill);
end

canB = snapshot(fname = "Frame001.svg");
setline(1);
for (pos, n) in itrTile
    sethue("white");
    pos += Point(250, 0) + Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2);
    box(pos, upScaleFctr, upScaleFctr, :stroke);
end

canC = snapshot(fname = "Frame002.svg");

cellTable = Table(tuOutSize[1], tuOutSize[2], upScaleFctr, upScaleFctr, Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2) + Point(250, 0));
setline(3);
sethue("cyan");
tableIdx = 264;
markcells(cellTable, getcells(cellTable, [tableIdx]));

canD = snapshot(fname = "Frame003.svg");

pxPos = Point(upScaleFctr + 10, tuCanvasSize[2] ÷ 2);
setline(3);
sethue("cyan");
box(pxPos, 2 * upScaleFctr, 2 * upScaleFctr, action = :stroke);
sethue(mS[refPxIdx]);
box(pxPos, 2 * upScaleFctr, 2 * upScaleFctr, action = :fill);

sethue("gray");
for (pos, n) in itrTile
    pos += Point(250, 0) + Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2);
    arrow(pxPos, pos; linewidth = 0.25);
end

canE = snapshot(fname = "Frame004.svg");

finish();
display(canB);
display(canC);
display(canD);
display(canE);



# placeimage(mU);
# finish();
# display(canA);

# mC = copy(mU);
# Drawing(tuCanvasSize[1], tuCanvasSize[2], :rec);
# placeimage(mC);
# sethue("white");
# setline(1);
# cellTable = Table(tuOutSize[1], tuOutSize[2], upScaleFctr, upScaleFctr, Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2));
# for (pos, n) in cellTable
#     box(pos, upScaleFctr, upScaleFctr, :stroke);
# end
# canB = snapshot(fname = "002TableImage.png");
# display(canB);

# setline(3);
# sethue("cyan");
# tableIdx = 
# markcells(cellTable, getcells(cellTable, [264]));
# canC = snapshot(fname = "003CellImage.png");
# finish();
# display(canC);

# canA = Drawing(tuCanvasSize[1], tuCanvasSize[2], :rec);
# placeimage(mC, Point(250, 0));
# setline(3);
# sethue("cyan");
# pxPos = Point(upScaleFctr + 10, tuCanvasSize[2] ÷ 2);
# box(pxPos, 2 * upScaleFctr, 2 * upScaleFctr, action = :stroke);
# sethue(mS[refPxIdx]);
# box(pxPos, 2 * upScaleFctr, 2 * upScaleFctr, action = :fill);
# sethue("gray");
# # Tiler is centering the area relative to (0, 0)
# itrTile = Tiler(tuSizeU[2], tuSizeU[1], tuOutSize[1], tuOutSize[2], margin = 0);
# setline(1);
# for (pos, n) in itrTile
#     pos += Point(250, 0) + Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2);
#     box(pos, upScaleFctr, upScaleFctr, :stroke);
# end
# vIdxImg = [1, 10, 50, 100, 150, 250]
# global imgIdx = 3;
# for (pos, n) in itrTile
#     pos += Point(250, 0) + Point(tuSizeU[2] ÷ 2, tuSizeU[1] ÷ 2);
#     arrow(pxPos, pos; linewidth = 0.25);
#     if n ∈ vIdxImg
#         global imgIdx;
#         imgIdx += 1;
#         imgStr = @sprintf("%03dMatchImage.png", imgIdx);
#         display(snapshot(fname = imgStr));
#     end
# end
# imgIdx += 1;
# imgStr = @sprintf("%03dMatchImage.png", imgIdx);
# canD = snapshot(fname = imgStr);
# finish();
# display(canD);



