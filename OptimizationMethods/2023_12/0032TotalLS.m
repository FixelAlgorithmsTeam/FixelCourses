% Optimization Methods
% SVD & Linear Least Squares - Total Least Squares.
% Solving:
% $$ \arg \min_{x} 0.5 * || E, b ||_F^2 $$
% Subject to $$ (A + E) x = b + r $$
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
% TLS Works for affine function, for others it is trickier
modelOrder      = 1; %<! Polynomial Order / Degree
numSamples      = 75;
noiseStd        = 0.75;
noiseFctrGrid   = 0.05;

% Model


% Visualization



%% Generate / Load Data

vG      = linspace(0, 3, numSamples);
vG      = vG(:);
mG      = vG .^ (0:modelOrder);
vX      = 3 * randn(modelOrder + 1, 1); %<! Parameters (Ground truth)
vN      = noiseStd * randn(numSamples, 1);
vZ      = mG * vX; %<! Model Data
vB      = vZ + vN; %<! Measurements

% Model (Noisy grid)
vA      = vG + (noiseFctrGrid * rand(numSamples, 1));
vA      = sort(vA, 'ascend');
mA      = vA .^ (0:modelOrder);


%% Display Data

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vG, vZ, 'DisplayName', 'Model Data');
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

% Regularized LS
vXLS = mA \ vB;

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vG, vZ, 'DisplayName', 'Model Data');
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


%% Total LS

vXTLS = TotalLeastSquares(mA, vB);
% vXTLS(1) = (sum(vB) - sum(vA) * vXTLS(2)) / numSamples;


%% Display Analysis

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vG, vZ, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hSctrObj = scatter(vA, vB, 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
hLineObj = plot(vA, mA * vXLS, 'DisplayName', 'LS Solution');
set(hLineObj, 'LineWidth', lineWidthNormal);
% hLineObj = plot(vA, mA * vXTLS, 'DisplayName', 'Total LS Solution');
hLineObj = plot(vA, mA * vXTLS, 'DisplayName', 'Total LS Solution');
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


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

