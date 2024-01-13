% Optimization Methods
% SVD & Linear Least Squares - Weighted Least Squares for Trigonometric Polynomials.
% Solving:
% $$ \arg \min_{x} 0.5 * || A * x - b_i ||_2^2
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     06/01/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants


%% Parameters

% Data
numSamples  = 500;
dataFreq    = 0.05; %<! Keep below 0.5

vX = 0.5 + rand(2, 1); %<! Amplitude

% Visualization



%% Generate / Load Data

vT = 0:(numSamples - 1);
vT = vT(:);

mA = [sin(2 * pi * dataFreq * vT), cos(2 * pi * dataFreq * vT)];
vA = mA * vX;

% 
vW = 0.75 * rand(numSamples, 1);
vW = sort(vW, 'ascend');

% Generate a random orthonormal matrix
mT = rand(numSamples, numSamples);
[mQ, mR] = qr(mT);
% mQ = eye(numSamples);

% Weights
mW      = mQ.' * diag(vW) * mQ;
mWSqrt  = chol(mW);

% Colored Noise
mC = inv(mW);
mCSqrt = chol(mC, 'lower');

vN = mCSqrt * randn(numSamples, 1);
vB = vA + vN; %<! Data Samples


%% Display Data

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(1:numSamples, vA, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(1:numSamples, vB, 20, vW, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
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
% 1. Calculate the least squares solution to estimate the amplitude of the
% data given `mA` and `vY`. Name it `vXLS`.

%----------------------------<Fill This>----------------------------%
vXLS = mA \ vB;
%-------------------------------------------------------------------%


%% Weighted Least Squares Solution
% 1. Calculate the weighted least squares estimation of the amplitude of
% the data given `mA`, `mW`, `mWSqrt` and `vY`. Name it `vXWLS`.
% !! You may chose either by whitening or the direct method.

%----------------------------<Fill This>----------------------------%
% Whitening
vYY = mWSqrt * vB;
mAA = mWSqrt * mA;
vXWLS = mAA \ vYY;

% Direct
vXWLS = (mA.' * mW * mA) \ (mA.' * mW * vB);
%-------------------------------------------------------------------%


%% Display Analysis

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(1:numSamples, vA, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(1:numSamples, vB, 20, vW, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
hLineObj = plot(1:numSamples, mA * vXLS, 'DisplayName', 'Least Squares');
set(hLineObj, 'LineWidth', lineWidthNormal)
hLineObj = plot(1:numSamples, mA * vXWLS, 'DisplayName', 'Weighted Least Squares');
set(hLineObj, 'LineWidth', lineWidthNormal)
set(get(hA, 'Title'), 'String', {['Model, Noisy Samples and Estimators']}, 'FontSize', fontSizeTitle);
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

disp(['LS  Estimatior L2 Norm Error: ', num2str(norm(vXLS - vX))]);
disp(['WLS Estimatior L2 Norm Error: ', num2str(norm(vXWLS - vX))]);


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

