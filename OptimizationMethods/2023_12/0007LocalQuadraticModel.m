% Optimization Methods
% Convex Optimization - Smooth Optimization - Local Quadratic Model
% Interpolating sampled by a Quadratic Model.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     27/12/2023
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
% y_i = 0.5 * vX_i.' * mA * vX_i + vB.' * vX_i + c

% 1. Build the linear model matrix:
%    mH * vW = mY(:)
% 2. Solve the Least Squares model for the parameters.

%----------------------------<Fill This>----------------------------%
gridLength = length(vX);
numGridPts = gridLength * gridLength;
vX1 = repmat(vX, gridLength, 1);
vX2 = repelem(vX, gridLength);
mH = [ones(numGridPts, 1), vX1, vX2, vX1 .^ 2, vX2 .^ 2, vX1 .* vX2];
vW = mH \ mY(:);
%-------------------------------------------------------------------%



%% Find Extrema
% 1. Extract `mA`, `vB` and `valC` from the parameters.

%----------------------------<Fill This>----------------------------%
mA      = [2 * vW(4), vW(6); vW(6), 2 * vW(5)];
vB      = [vW(2); vW(3)];
valC    = vW(1);
%-------------------------------------------------------------------%

% Calculating the Estimated Values
% Vectorized
mX = [vX1, vX2];
vYEst = 0.5 * diag(mX * mA * mX.') + mX * vB + valC;

% Loop
vYEst = zeros(numGridPts, 1);
for ii = 1:numGridPts
    vYEst(ii) = 0.5 * mX(ii, :) * mA * mX(ii, :).' + mX(ii, :) * vB + valC;
end

% Linear Model
% vYEst = mH * vW;

mYEst = reshape(vYEst, gridLength, gridLength);

% Find Arg Max

%----------------------------<Fill This>----------------------------%
vXMax = -mA \ vB;
%-------------------------------------------------------------------%


%% Display Results

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
set(hA, 'NextPlot', 'add');
plot(1:10);
hSurfObj = surf(vX, vX, mY);
set(hSurfObj, 'EdgeColor', 'none', 'FaceAlpha', 0.35);
hSurfObj = surf(vX, vX, mYEst);
set(hA, 'View', [-37.5, 30]);
set(hSurfObj, 'EdgeColor', 'none', 'FaceAlpha', 0.85, 'FaceColor', 'r');
set(get(hA, 'Title'), 'String', {['Local Quadratic Approximation']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x_1']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['x_2']}, 'FontSize', fontSizeAxis);
set(get(hA, 'ZLabel'), 'String', {['y_i']}, 'FontSize', fontSizeAxis);

% hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

if(generateFigures == ON)
    set(hF, 'Color', 'none');
    set(hA, 'Color', 'none');
    % set(hLegend, 'Color', 'none');
    % set(hLegend, 'TextColor', 'white');
    % set(hLegend, 'LineWidth', 3);
    % set(get(hA, 'Title'), 'Color', 'white');
    % set(hA, 'GridColor', 'white', 'MinorGridColor', 'white');
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
    % print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.svg'], '-vector', '-dsvg');
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

