% Optimization Methods
% Convex Optimization - Smooth Optimization - Logistic Regression
% Optimizing a classifier using the Logistic Regression model.
% The model is given by:
% $$ arg min_w 0.5 * || sigma(X * w) - y ||_2^2 $$
% Where sigma(x) is a scaled and translated version of the Sigmoid
% function.
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
dataFileName = 'LogRegData.mat';

% Numerical Differentiation
diffMode    = DIFF_MODE_CENTRAL;
errTol      = 1e-6;

% Solver
stepSizeMode    = STEP_SIZE_MODE_ADAPTIVE;
stepSize        = 0.01;
numIterations   = 100;

% Visualization
numGridPts = 501;
vLim = [-2, 2]; %<! Boundaries for Xlim / Ylim


%% Generate / Load Data
% The data is:
% - mX: The coordinates of the samples in a 2D grid.
% - vY: The labels of the samples in the range {-1, 1}.

% Symmetric Matrix
sData = load(dataFileName);
mX = sData.mX; %<! The data
mX(:, 1) = mX(:, 1) - 0.5;
vY = double(sData.vY); %<! The labels
vY(vY == 0) = -1;

numSamples = size(mX, 1);

% Analysis
mW      = zeros(size(mX, 2) + 1, numIterations); %<! Initialization is the zero vector
vObjVal = zeros(numIterations, 1);

% Visualization
[mGx, mGy] = meshgrid(linspace(-2, 2, numGridPts));
mG = cat(2, -ones(numGridPts ^ 2, 1), mGx(:), mGy(:));


%% Display Data

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
mColorOrder = get(hA, 'ColorOrder');
mC = mColorOrder(1:2, :); %<! Binary
mC = repelem(mC, 2, 1); %<! Avoids a bug in `contouf()` with binary colormap
set(hA, 'Colormap', mC);

PlotClassData(mX, vY, 'plotTitle', 'Binary Classification Data', 'hA', hA);
set(hA, 'XLim',vLim, 'YLim', vLim);
set(get(hA, 'XLabel'), 'String', {['x_1']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['x_2']}, 'FontSize', fontSizeAxis);


%% Arrange Data
% The data model is linear function of the coordinates.  
% Hence the model matrix, per row (Sample) is built as:
% [1, x1, x2]

mX = cat(2, -ones(numSamples, 1), mX);
hObjFun = @(vW) 0.5 * sum((CalcSigmoidFun(mX * vW) - vY) .^ 2);


%% Gradient Function
% 1. Derive the gradient of the objective function.
% 2. Implement it in `CalcGradLogRegFun()`.
% 2. Per `vX`, set the `hObjFunGrad()`.

%----------------------------<Fill This>----------------------------%
hObjFunGrad = @(vX) CalcGradLogRegFun(vX, mX, vY);
%-------------------------------------------------------------------%

vX = randn(size(mW, 1), 1);

vG = CalcFunGrad(vX, hObjFun, diffMode);
assertCond = norm(hObjFunGrad(vX) - vG, 'inf') <= (errTol * norm(vG));
assert(assertCond, 'The analytic calculation deviation exceeds the threshold %f', errTol);

disp(['The analytic implementation is verified']);


%% Gradient Descent
% 1. Update `GradientDescent()` with adaptive step size logic (Backtracking like).

mW = GradientDescent(mW, hObjFunGrad, stepSizeMode, stepSize, hObjFun);


%% Display Results

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
mColorOrder = get(hA, 'ColorOrder');
mC = mColorOrder(1:2, :); %<! Binary
mC = repelem(mC, 2, 1); %<! Avoids a bug in `contouf()` with binary colormap
set(hA, 'Colormap', mC);

PlotClassData(mX(:, 2:3), vY, 'plotTitle', 'Binary Classification Data', 'hA', hA);
set(hA, 'XLim', vLim, 'YLim', vLim);
set(hA, 'DataAspectRatio', [1, 1, 1]);
set(get(hA, 'XLabel'), 'String', {['x_1']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['x_2']}, 'FontSize', fontSizeAxis);


ii = 1;
vZ = sign(mG * mW(:, ii));
mZ = reshape(vZ, numGridPts, numGridPts);
mZ(mZ == -1) = 0;
hQvrObj = quiver(0, (mW(1, ii) / mW(3, ii)), mW(2, ii) / norm(mW(2:3, ii)), mW(3, ii) / norm(mW(2:3, ii)), 'LineWidth', lineWidthThick);
set(hQvrObj, 'Color', 'red', 'DisplayName', 'vW');
[~, hCntrObj] = contourf(mGx, mGy, mZ, 'FaceAlpha', 0.15, 'DisplayName', 'Decision Boundary');

ClickableLegend();

for ii = 1:numIterations
    vZ = sign(mG * mW(:, ii));
    mZ = reshape(vZ, numGridPts, numGridPts);
    mZ(mZ == -1) = 0;
    set(hQvrObj, 'XData', 0, 'YData', (mW(1, ii) / mW(3, ii)), 'UData', mW(2, ii) / norm(mW(2:3, ii)), 'VData',  mW(3, ii) / norm(mW(2:3, ii)));
    set(hCntrObj, 'ZData', mZ);
    clsAcc = mean(sign(mX * mW(:, ii)) == vY);
    set(get(hA, 'Title'), 'String', {['Logistic Regression: Iteration ', num2str(ii, '%04d'), ', Accuracy ', num2str(100 * clsAcc, '%0.2f'), '%']}, 'FontSize', fontSizeTitle);

    drawnow();
    pause(0.002);
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

