% Optimization Methods
% Loss Functions - DFT Super Resolution by L1 Norm.
% Solving:
% $$ \arg \min_{x} 0.5 * || F x - f ||_F^2 + λ * || x ||_1 $$
% Where $F$ is the inverse DFT matrix.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     13/01/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;

%% Constants


%% Parameters

% Data
samplingFreq        = 50; %<! [Hz]
observationPeriod   = 1; %<! [Sec]

% Guideline: Easy to resolve for |f1 - f2| > (1 / T) where T is the
% observation interval.
sineFreq1 = 5;

% Frequency difference
vD = [0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5];
vD = vD(:);

vF = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00];
% vF = [0.25, 0.5, 0.75];
vF = vF(:);

% Model
superResolutionFactor = 64;

cSuperResolutionDft = {@(vY, mF, paramLambda) fft(vY, size(mF, 2)), ...
    @(vY, mF, paramLambda) SuperResolutionDftL1(vY, mF, paramLambda)};

cSuperResolutionDftString = {['Zero Padding'], ['L1 Sparse Model']};

% Visualization



%% Generate / Load Data

vT = linspace(0, observationPeriod, (observationPeriod * samplingFreq) + 1);
vT = vT(:);
vT = vT(1:end - 1);

vY1 = sin(2 * pi * sineFreq1 * vT);

% The Model Matrix
numSamples  = size(vT, 1);
numFreqBins = superResolutionFactor * numSamples;
mF          = exp(2i * pi * [0:(numSamples - 1)].' * [0:(numFreqBins - 1)] / numFreqBins) / numFreqBins;


%% Display Data

figureIdx = figureIdx + 1;

hF = figure('Position', figPosX2Large);

hTiledLayoutObj = tiledlayout(length(vD), 1);
set(hTiledLayoutObj, 'TileSpacing', 'compact', 'Padding', 'compact');
set(get(hTiledLayoutObj, 'XLabel'), 'String', 'Frequency [Hz]', 'FontSize', 12);
set(get(hTiledLayoutObj, 'YLabel'), 'String', 'Amplitude', 'FontSize', 12);

for ii = 1:length(vD)
    vY2 = sin(2 * pi * (sineFreq1 + vD(ii)) * vT);
    vY = vY1 + vY2;
    hAxes = nexttile(hTiledLayoutObj);
    plotTitle = ['F_1 = ', num2str(sineFreq1), ' [Hz], F_2 = ', num2str(sineFreq1 + vD(ii)), ' [Hz]'];
    hLineObject = PlotDft(vY, samplingFreq, 'singleSideFlag', ON, 'plotTitle', plotTitle, 'logScaleFlag', OFF, 'plotXYLabelFlag', OFF);
end

if(generateFigures == ON)
    set(hF, 'Color', 'none');
    vHA = findobj(hF, '-depth', 1, 'type', 'axes');
    for ii = 1:length(vHA)
        hA = vHA(ii);
        set(hA, 'Color', 'none');
        set(get(hA, 'Title'), 'Color', 'white');
        set(hA, 'GridColor', 'white', 'MinorGridColor', 'white');  
        set(hA, 'XColor', [0.75, 0.75, 0.75]);
        set(hA, 'YColor', [0.75, 0.75, 0.75]);
        set(get(hA, 'XLabel'), 'Color', [0.75, 0.75, 0.75]);
        set(get(hA, 'YLabel'), 'Color', [0.75, 0.75, 0.75]);
    end
    vHL = findobj(hF, '-depth', 1, 'type', 'legend');
    for ii = 1:length(vHL)
        hL = vHL(ii);
        set(hL, 'Color', 'none');
        set(hL, 'TextColor', 'white');
        set(hL, 'LineWidth', 3);
    end
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end


%% Zero Padding Solution




%% Super Resolutuion by L1 Solution



%% Display Analysis

numPlots    = size(vF, 1);
numMethods  = length(cSuperResolutionDft);

paramLambda = 0.002;

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosX2Large);

hTiledLayoutObj = tiledlayout(numPlots, numMethods);
set(hTiledLayoutObj, 'TileSpacing', 'compact', 'Padding', 'compact');
set(get(hTiledLayoutObj, 'XLabel'), 'String', 'Frequency [Hz]', 'FontSize', 12);
set(get(hTiledLayoutObj, 'YLabel'), 'String', 'Amplitude [dB]', 'FontSize', 12);

for ii = 1:numPlots
    vY2 = sin(2 * pi * (sineFreq1 + vF(ii)) * vT);
    vY = vY1 + vY2;
    for jj = 1:numMethods
        vX = cSuperResolutionDft{jj}(vY, mF, paramLambda);
        hA = nexttile(hTiledLayoutObj);
        if(jj == 1)
            PlotDft(vX, samplingFreq, 'applyDft', OFF, 'plotTitle', [cSuperResolutionDftString{jj}, ' F_1 = ', num2str(sineFreq1), ' [Hz], F_2 = ', num2str(sineFreq1 + vF(ii)), ' [Hz]'], 'plotXYLabelFlag', OFF);
        else
            PlotDft(vX, samplingFreq, 'applyDft', OFF, 'plotTitle', cSuperResolutionDftString{jj}, 'plotXYLabelFlag', OFF);
        end
        set(hA, 'XLim', [3, 7]);
    end
end
% 
% if(generateFigures == ON)
%     set(hF, 'Color', 'none');
%     vHA = findobj(hF, '-depth', 1, 'type', 'axes');
%     for ii = 1:length(vHA)
%         hA = vHA(ii);
%         set(hA, 'Color', 'none');
%         set(get(hA, 'Title'), 'Color', 'white');
%         set(hA, 'GridColor', 'white', 'MinorGridColor', 'white');  
%         set(hA, 'XColor', [0.75, 0.75, 0.75]);
%         set(hA, 'YColor', [0.75, 0.75, 0.75]);
%         set(get(hA, 'XLabel'), 'Color', [0.75, 0.75, 0.75]);
%         set(get(hA, 'YLabel'), 'Color', [0.75, 0.75, 0.75]);
%     end
%     vHL = findobj(hF, '-depth', 1, 'type', 'legend');
%     for ii = 1:length(vHL)
%         hL = vHL(ii);
%         set(hL, 'Color', 'none');
%         set(hL, 'TextColor', 'white');
%         set(hL, 'LineWidth', 3);
%     end
%     exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
% end


%% Auxiliary Functions

function [ vX ] = SuperResolutionDftL1( vY, mF, paramLambda )

% numIterations = 25000;
% mX = zeros(size(mF, 2), numIterations);
% 
% hGradFun = @(vX) mF' * (mF * vX - vY);
% % Complex sign function: `sign(vY) = exp(1i * angle(vY));`.
% % MATLAB's `sign()` function supports complex numbers as above.
% hProxFun = @(vY, paramLambda) max(abs(vY) - paramLambda, 0) .* exp(1i * angle(vY));
% stepSize = 2.00;
% 
% mX = ProxGradientDescentAccel(mX, hGradFun, hProxFun, stepSize, paramLambda);
% 
% vX = mX(:, numIterations);

numFreqBins = size(mF, 2);

cvx_begin quiet
    variable vX(numFreqBins) complex
    minimize( 0.5 * sum_square_abs((mF * vX) - vY) + paramLambda * norm(vX, 1) )
cvx_end

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

