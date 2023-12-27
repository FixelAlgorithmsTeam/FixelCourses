% Optimization Methods
% Convex Optimization - Smooth Optimization - Local Quadratic Model
% Findnig the max and arg max of the model.
% The model is given by:
% $$ 0.5 * || A x - y ||_2^2 $$
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     10/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;

%% Constants

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;


%% Parameters

% Data
gridRadius = 4;

kernelMean  = -0.2;
kernelStd   = 1.5;


%% Generate / Load Data

% Symmetric Matrix
vX = -gridRadius:gridRadius;
vX = vX(:);
vY = exp(-0.5 * ((vX - kernelMean) .^ 2) / (kernelStd * kernelStd) );
mY = vY * vY.';

%% Display Data

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
hSurfObj = surf(vX, vX, mY);
set(hSurfObj, 'EdgeColor', 'none');
set(get(hA, 'Title'), 'String', {['Data Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x_1']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['x_2']}, 'FontSize', fontSizeAxis);
set(get(hA, 'ZLabel'), 'String', {['y_i']}, 'FontSize', fontSizeAxis);


%% Build Linear Model & Estimate Parameters
% if mA was known then the model is:
% y_i = vX_i.' * mA * vX_i * vB.' * vX_i + c
% Think of calculating vY in a vectorized way vs. in a loop.

% Build the model as:
% mH * vW = vZ

%----------------------------<Fill This>----------------------------%
gridLength = length(vX);
numGridPts = gridLength * gridLength;
vX1 = repmat(vX, gridLength, 1);
vX2 = repelem(vX, gridLength);
mS = [ones(numGridPts, 1), vX1, vX2, vX1 .^ 2, vX2 .^ 2];
vW = mS \ mY(:);
%-------------------------------------------------------------------%



%% Find Extrema
% 1. Extract `mA`, `vB` and `valC` from the parameters.
% 2. 


%% Analysis

% Reference Solution
vXRef       = mA \ vB;
objValRef   = hObjFun(vXRef);

for ii = 1:numIterations
    vObjVal(ii) = hObjFun(mX(:, ii));
end

vObjVal = 20 * log10(abs(vObjVal - objValRef) / max(abs(objValRef), sqrt(eps())));


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numIterations, vObjVal, 'DisplayName', 'Adaptive Step Size');
set(hLineObj, 'LineWidth', lineWidthNormal);

set(get(hAxes, 'Title'), 'String', {['Objective Function Convergence with Adaptive Step Size']}, 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Index']}, 'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Relative Error [dB]']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

