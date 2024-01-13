% Optimization Methods
% SVD & Linear Least Squares - Sequential Least Squares.
% Solving:
% $$ \arg \min_{x} 0.5 * || A * x - y ||_2^2 + α ||x||_2^2 $$
% Given new samples, solve the problem sequentially.
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
modelOrder      = 3; %<! Polynomial Order / Degree
numSamples      = 25;
numSamplesBatch = 5;
noiseStd        = 2;

% Model
paramAlpha = 1;


% Visualization



%% Generate / Load Data

vA      = linspace(0, 3, numSamples);
vA      = vA(:);
mA      = vA .^ (0:modelOrder);
vX      = 3 * randn(modelOrder + 1, 1); %<! Parameters (Ground truth)
vN      = noiseStd * randn(numSamples, 1);
vZ      = mA * vX; %<! Model Data
vB      = vZ + vN; %<! Measurements


%% Display Data

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vA, vZ, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(vA, vB, 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
set(get(hA, 'Title'), 'String', {['Model and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Value']}, 'FontSize', fontSizeAxis);
ClickableLegend();

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

%% Least Squares Solution
% The reference for the sequential results.

% Regularized LS
vXLS = (mA.' * mA + paramAlpha * eye(modelOrder + 1)) \ (mA.' * vB);

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vA, vZ, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hSctrObj = scatter(vA, vB, 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
hLineObj = plot(vA, mA * vXLS, 'DisplayName', 'LS Solution');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hA, 'Title'), 'String', {['Model and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Value']}, 'FontSize', fontSizeAxis);
ClickableLegend();

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


%% Sequetial LS

mXSLS = zeros(size(vX, 1), numSamples - numSamplesBatch + 1);

mR = pinv(mA(1:numSamplesBatch, :).' *  mA(1:numSamplesBatch, :) + paramAlpha * eye(modelOrder + 1));
vXSLS = mR * (mA(1:numSamplesBatch, :).' * vB(1:numSamplesBatch));

jj = 1;
mXSLS(:, jj) = vXSLS;

for ii = (numSamplesBatch + 1):numSamples
    jj = jj + 1;
    valY    = vB(ii);
    vAs     = mA(ii, :).';
    [vXSLS, mR] = SequnetialLeastSquares(vXSLS, valY, vAs, mR);
    mXSLS(:, jj) = vXSLS;
end


%% Display Analysis


minVal = min(vB);
maxVal = max(vB);

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA   = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vA, vZ, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hSctrObj = scatter(vA, vB, 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
hLineObj = plot(vA, mA * vXLS, 'DisplayName', 'LS Solution');
set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
hLineObj = plot(vA, mA * mXSLS(:, 1), 'DisplayName', 'Sequential LS Solution');
set(hLineObj, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
set(hA, 'YLim', [10 * floor(minVal / 10), 10 * ceil(maxVal / 10)]);
set(hA, 'XLim', [min(vA), max(vA)]);
set(get(hA, 'Title'), 'String', {['Sequential Least Squares Estimation vs. Batch Least Squares Estimation'], ['Sequential Estimation Based on Batch Mode of ', num2str(numSamplesBatch), ' First Samples and ', num2str(0) ' Sequential Samples']}, ...
    'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Smaple Value']}, ...
    'FontSize', fontSizeAxis);
hLegend = ClickableLegend();

pause(1);

jj = 1;
for ii = (numSamplesBatch + 1):numSamples
    set(hLineObj, 'YData', mA * mXSLS(:, jj + 1));
    set(get(hA, 'Title'), 'String', {['Sequential Least Squares Estimation vs. Batch Least Squares Estimation'], ['Sequential Estimation Based on Batch Mode of ', num2str(numSamplesBatch), ' First Samples and ', num2str(jj) ' Sequential Samples']}, ...
    'FontSize', fontSizeTitle);
    pause(1);
    jj = jj + 1;
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


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

