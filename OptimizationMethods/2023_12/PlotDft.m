function [ hF, hA, hL ] = PlotDft( mX, samplingFrequency, sPlotDftOpt )
% ----------------------------------------------------------------------------------------------- %
% [ hF, hA, hL ] = PlotDft( mX, samplingFrequency, sPlotDftOpt )
%   Plots the DFT of the input signal (Or matrix of signals as columns).
% Input:
%   - mX                -   Input Signals Matrix.
%                           Structure: Matrix (numSamples x numSignals).
%                           Type: 'Single' / 'Double'.
%                           Range: (-inf, inf).
%   - samplingFrequency -   Sampling Frequency.
%                           The sampling frequency of the data in Hz.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: (0, inf).
%   - singleSideFlag    -   Single Side Flag.
%                           If set to 1 (ON) only single side will be
%                           displayed (Assuming Real Data).
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - logScaleFlag      -   logScaleFlag
%                           If set to 1 data will be displayed in a log
%                           scale.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - normalizDataFlag  -   normalizDataFlag.
%                           If set to 1 the peak magnitude will be
%                           normalized to have value of 1 (0 on logarithmic
%                           scale).
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - openFig           -   Open a Figure.
%                           If set to 1 a new figure will be used.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - numFreqBins       -   Number of Frequency Bins.
%                           The number of frequency bins in the frequency
%                           domain.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {numSamples, numSamples + 1, ...}.
%   - applyDft          -   Apply DFT Transform.
%                           If set to 1 the DFT Transform (`fft()`) will be
%                           applied on the input data. If set to 0 it is
%                           assumed the data is already in the frequency
%                           domain.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - removeDc          -   Remove DC Component.
%                           If set to 1 the input data will have its DC
%                           (Mean value) removed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - plotTitleFlag     -   Plot Title Flag.
%                           If set to 1 the plot title will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - plotXYLabelFlag   -   Plot X / Y Labels Flag.
%                           If set to 1 the plot X / Y labels will be
%                           displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
%   - plotLegendFlag    -   Plot Legend Flag.
%                           If set to 1 the plot legend will be displayed.
%                           Structure: Scalar.
%                           Type: 'Single' / 'Double'.
%                           Range: {0, 1}.
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
%   - hL                -   Line Object Handler.
%                           The line object handler of the output plot.
%                           Structure: Scalar.
%                           Type: NA.
%                           Range: NA.
% References:
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
%   Release Notes:
%   -   1.3.001     18/07/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   Fixed issue with multiple signals when the legend string
%           `plotLegend` isn't given.
%       *   Fixed issue with `singleSideFlag` in case the samplig frequency
%           is a fraction.
%       *   Added `removeDc` option.
%   -   1.3.000     31/03/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   Added the `plotLegend` and `plotLegendFlag` parameters.
%   -   1.2.000     16/01/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   Added the `numFreqBins` parameter.
%       *   Added the `applyDft` parameter.
%       *   Added the `plotTitleFlag` and `plotXYLabelFlag` parameters.
%   -   1.1.000     09/01/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   Using struct of parameters.
%       *   Added option to set whether a new figure should be generated.
%   -   1.0.001     04/01/2021  Royi Avital     RoyiAvital@yahoo.com
%       *   Fixed a bug when `logScaleFlag = OFF`.
%       *   Fixed a bug with the frequency grid.
%   -   1.0.000     24/09/2020  Royi Avital     RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments
    mX (:, :) {mustBeNumeric}
    samplingFrequency (1, 1) {mustBeNumeric, mustBeReal, mustBePositive}
    sPlotDftOpt.singleSideFlag (1, 1) {mustBeMember(sPlotDftOpt.singleSideFlag, [0, 1])} = 1
    sPlotDftOpt.logScaleFlag (1, 1) {mustBeMember(sPlotDftOpt.logScaleFlag, [0, 1])} = 1
    sPlotDftOpt.normalizDataFlag (1, 1) {mustBeMember(sPlotDftOpt.normalizDataFlag, [0, 1])} = 0
    sPlotDftOpt.openFig (1, 1) {mustBeMember(sPlotDftOpt.openFig, [0, 1])} = 0
    sPlotDftOpt.plotTitle (1, :) string = {['DFT']}
    sPlotDftOpt.plotLegend (1, :) string = {}
    sPlotDftOpt.numFreqBins (1, 1) {mustBeNumeric, mustBeReal, mustBePositive, mustBeInteger} = size(mX, 1)
    sPlotDftOpt.applyDft (1, 1) {mustBeMember(sPlotDftOpt.applyDft, [0, 1])} = 1
    sPlotDftOpt.removeDc (1, 1) {mustBeMember(sPlotDftOpt.removeDc, [0, 1])} = 0
    sPlotDftOpt.plotTitleFlag (1, 1) {mustBeMember(sPlotDftOpt.plotTitleFlag, [0, 1])} = 1
    sPlotDftOpt.plotXYLabelFlag (1, 1) {mustBeMember(sPlotDftOpt.plotXYLabelFlag, [0, 1])} = 1
    sPlotDftOpt.plotLegendFlag (1, 1) {mustBeMember(sPlotDftOpt.plotLegendFlag, [0, 1])} = 0
end

FALSE   = 0;
TRUE    = 1;

OFF = 0;
ON  = 1;

% Unpacking parameters struct
singleSideFlag      = sPlotDftOpt.singleSideFlag;
logScaleFlag        = sPlotDftOpt.logScaleFlag;
normalizDataFlag    = sPlotDftOpt.normalizDataFlag;
openFig             = sPlotDftOpt.openFig;
plotLegend          = sPlotDftOpt.plotLegend;
plotTitle           = sPlotDftOpt.plotTitle;
numFreqBins         = sPlotDftOpt.numFreqBins;
applyDft            = sPlotDftOpt.applyDft;
removeDc            = sPlotDftOpt.removeDc;
plotTitleFlag       = sPlotDftOpt.plotTitleFlag;
plotXYLabelFlag     = sPlotDftOpt.plotXYLabelFlag;
plotLegendFlag      = sPlotDftOpt.plotLegendFlag;

if(size(mX, 1) == 1)
    mX = mX(:);
end

numSamples  = size(mX, 1);

if(removeDc == ON)
    mX = mX - mean(mX, 1);
end

if(applyDft == ON)
    numFreqBins = max(numFreqBins, numSamples);
    mXK = fft(mX, numFreqBins, 1);
else
    numFreqBins = numSamples;
    mXK = mX;
end

% The frequency grid
vK = linspace(0, samplingFrequency, numFreqBins + 1);
vK = vK(1:numFreqBins);

yLabelString = 'Amplitude';

if(normalizDataFlag == ON)
    mXK = mXK ./ max(abs(mXK), [], 1);
    yLabelString = [yLabelString, ' (Normalized)'];
end

if(logScaleFlag == ON)
    mXK             = 20 * log10(abs(mXK));
    yLabelString    = [yLabelString, ' [dB]'];
else
    mXK = abs(mXK);
end

if(openFig == ON)
    hF = figure('Position', [100, 100, 800, 600]);
    hA = axes();
else
    hF = gcf();
    hA = gca();
end
hL = plot(vK, mXK);
if(plotTitleFlag == ON)
    set(get(hA, 'Title'), 'String', plotTitle, 'FontSize', 14);
end
if(plotXYLabelFlag == ON)
    set(get(hA, 'XLabel'), 'String', 'Frequency [Hz]', 'FontSize', 12);
    set(get(hA, 'YLabel'), 'String', yLabelString, 'FontSize', 12);
end
if(plotLegendFlag == ON)
    ClickableLegend(plotLegend);
end

if(singleSideFlag == ON)
    set(hA, 'XLim', [0, samplingFrequency / 2]);
end


end

