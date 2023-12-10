function [ hF, hA ] = PlotClassData( mX, vY, sPlotOpt )
% ----------------------------------------------------------------------------------------------- %
% [ hF, hA ] = PlotClassData( mX, vY, sPlotOpt )
%   Plots 2D binary classification data.
% Input:
%   - mI                -   Input Image.
%                           Structure: Matrix (numRows x numCols x numChannels).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - zoomLevel         -   Zoom Level.
%                           The zoom level of the image.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - mAlphaData        -   The Alpha Channel of the Image.
%                           Structure: Matrix (numRows x numCols).
%                           Type: 'Single' / 'Double'.
%                           Range: [0, 1].
%   - showAxis          -   Show Axis.
%                           If set to 1 axis will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - showTicks         -   Show Axis Ticks.
%                           If set to 1 the axis ticks will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - marginSize        -   Margin Size.
%                           The margin around the axes in pixels.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1, 2, ...}.
%   - hA                -   Axes Handler.
%                           The axes handler to use for the plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - openFig           -   Open a Figure.
%                           If set to 1 a new figure will be used.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - plotTitle         -   The Plot Title String.
%                           The string to plot as title. If empty, no title
%                           will be displayed.
%                           Structure: String.
%                           Type: String.
%                           Range: NA.
% Output:
%   - hF                -   Figure Handler.
%                           The figure handler of the output plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - hA                -   Axes Handler.
%                           The axes handler of the output plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
%   - hImgObj           -   Image Object Handler.
%                           The image object handler.
%                           Structure: Scalar.
%                           Type: Handler / Object.
%                           Range: NA.
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
%   Release Notes:
%   -   1.0.000     24/11/2023  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mX (:, 2) {mustBeNumeric, mustBeReal}
    vY (:, 1) {mustBeNumeric, mustBeReal, mustBeInteger}
    sPlotOpt.markerSize (1, 1) {mustBeNonnegative, mustBeInteger} = 50
    sPlotOpt.marginSize (1, 1) {mustBeNonnegative, mustBeInteger} = 50
    sPlotOpt.hA (1, 1) {mustBeA(sPlotOpt.hA, ["matlab.graphics.axis.Axes", 'double'])} = 0
    sPlotOpt.openFig (1, 1) {mustBeMember(sPlotOpt.openFig, [0, 1])} = 0
    sPlotOpt.plotTitle (1, :) string = {['']}
    sPlotOpt.showLegend (1, :) {mustBeMember(sPlotOpt.showLegend, [0, 1])} = 0
end

arguments(Output)
    hF (1, 1) {mustBeA(hF, ["matlab.ui.Figure", "double"])}
    hA (1, 1) {mustBeA(hA, ["matlab.graphics.axis.Axes", "double"])}
end

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

% Unpacking parameters struct
markerSize  = sPlotOpt.markerSize;
marginSize  = sPlotOpt.marginSize;
hA          = sPlotOpt.hA;
openFig     = sPlotOpt.openFig;
plotTitle   = sPlotOpt.plotTitle;
showLegend  = sPlotOpt.showLegend;

numSamples = size(mX, 1);

vC          = unique(vY);
numClass    = length(vC);

if(openFig == TRUE)
    hF = figure('Position', [100, 100, 2 * marginSize + numColsEff, 2 * marginSize + numRowsEff]);
    hA = axes(hF, 'Units', 'pixels', 'Position', [marginSize + 1, marginSize + 1, numColsEff, numRowsEff]);
elseif(isa(hA, 'matlab.graphics.axis.Axes'))
    hF = ancestor(hA, 'figure', 'toplevel');
else
    hF = gcf(); %<! Opens if no Figure is supplied
    hA = gca();
end

mColorOrder = get(hA, 'ColorOrder');
numColors = size(mColorOrder, 1);

set(hA, 'NextPlot', 'add');

for ii = 1:numClass
    vIdx = vY == vC(ii);
    hSctObj = scatter(mX(vIdx, 1), mX(vIdx, 2), 'filled', 'DisplayName', ['Class ', num2str(vC(ii))]);
    set(hSctObj, 'MarkerFaceColor', mColorOrder(mod(ii - 1, numColors) + 1, :), 'SizeData', markerSize);
end

if(isempty(plotTitle) == FALSE)
    set(get(hA, 'Title'), 'String', plotTitle);
end

if(showLegend)
    ClickableLegend();
end


end

