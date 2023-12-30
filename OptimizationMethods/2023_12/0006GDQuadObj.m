% Optimization Methods
% Convex Optimization - Smooth Optimization - Quadratic Objective
% Solving a quadratic optimization problem using Gradient Descent.
% The model is given by:
% $$ 0.5 * x^T A x + b^t x $$
% Where A is a PSD matrix.
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
numRows = 20;
numCols = numRows;

% Numerical Differntiation
diffMode    = DIFF_MODE_CENTRAL;
errTol      = 1e-6;

% Solver
stepSizeMode    = STEP_SIZE_MODE_CONSTANT;
vStepSize       = [0.01; 0.001; 0.0001; 0.00001];
numIterations   = 50000;


%% Generate / Load Data

% Symmetric Matrix
mA = randn(numRows, numCols);
mA = mA.' * mA + (0.05 * eye(numRows));
mA = mA + mA.';

vB = randn(numRows, 1);

hObjFun = @(vX) 0.5 * (vX.' * mA * vX) + (vB.' * vX);

% Analysis
numStepSize = length(vStepSize);
tX = zeros(numCols, numIterations, numStepSize); %<! Initialization is the zero vector
mObjVal = zeros(numIterations, numStepSize);


%% Gradient Function
% 1. Derive the gradient of the objective function.
% 2. Implement it as a function handler.

%----------------------------<Fill This>----------------------------%
hObjFunGrad = @(vX) (mA * vX) + vB;
%-------------------------------------------------------------------%

vX = randn(numCols, 1);

vG = CalcFunGrad(vX, hObjFun, diffMode);
assertCond = norm(hObjFunGrad(vX) - vG, 'inf') <= (errTol * norm(vG));
assert(assertCond, 'The analytic calculation deviation exceeds the threshold %f', errTol);

disp(['The analytic implementation is verified']);



%% Gradient Descent
% 1. Create a function called `GradientDescent()` according to the template.

for kk = 1:numStepSize
    stepSize = vStepSize(kk);
    tX(:, :, kk) = GradientDescent(tX(:, :, kk), hObjFunGrad, stepSizeMode, stepSize);
end


%% Analysis

% Reference Solution
vXRef       = -mA \ vB;
objValRef   = hObjFun(vXRef);

for jj = 1:numStepSize
    for ii = 1:numIterations
        mObjVal(ii, jj) = hObjFun(tX(:, ii, jj));
    end
end

mObjVal = 20 * log10(abs(mObjVal - objValRef) / max(abs(objValRef), sqrt(eps())));


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numIterations, mObjVal);
for ii = 1:numStepSize
    set(hLineObj(ii), 'DisplayName', ['Step Size: ', num2str(vStepSize(ii), '%0.5f')]);
end
set(hLineObj, 'LineWidth', lineWidthNormal);

set(get(hAxes, 'Title'), 'String', {['Objective Function Convergence by Step Size']}, 'FontSize', fontSizeTitle);
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

