% Optimization Methods
% Convex Optimization - Smooth Optimization - Cooridnate Descent
% Solving a least squares porblem using Coordinate Descent.
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
numRows = 20;
numCols = numRows;

% Numerical Differntiation
diffMode    = DIFF_MODE_CENTRAL;
errTol      = 1e-6;

% Solver
stepSizeMode    = STEP_SIZE_MODE_ADAPTIVE;
stepSize        = 0.01;
numIterations   = 10000;


%% Generate / Load Data

% Symmetric Matrix
mA = randn(numRows, numCols);
mA = mA.' * mA + (0.95 * eye(numRows));
mA = mA + mA.';

vB = randn(numRows, 1);

hObjFun = @(vX) 0.5 * sum((mA * vX - vB) .^ 2);

% Analysis
mX = zeros(numCols, numIterations); %<! Initialization is the zero vector
vObjVal = zeros(numIterations, 1);


%% Gradient Function
% 1. Derive the gradient of the objective function (Per component).
% 2. Implement it as a function handler.

mAA = mA.' * mA;

%----------------------------<Fill This>----------------------------%
hObjFunGrad = @(vX, jj) mAA(jj, :) * vX - mA(:, jj).' * vB;
%-------------------------------------------------------------------%

vX = randn(numCols, 1);

ii  = randi(numCols);
vEi = zeros(numCols, 1);
vEi(ii) = 1;

vG = CalcFunGrad(vX, hObjFun, diffMode);
assertCond = norm(hObjFunGrad(vX, ii) - vEi.' * vG, 'inf') <= (errTol * norm(vG));
assert(assertCond, 'The analytic calculation deviation exceeds the threshold %f', errTol);

disp(['The analytic implementation is verified']);



%% Coordinate Descent
% 1. Create a function called 'GradientDescent()` according to the template.
% 2. Implemnt the adaptive step size logic (Backtracking like).

mX = CoordinateDescent(mX, hObjFunGrad, stepSizeMode, stepSize, hObjFun);


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

