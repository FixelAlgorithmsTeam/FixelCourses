% Optimization Methods
% Convex Optimization - Constraint Optimization - LS with Unit Simple Constraints
% Estimating a Low Pass Filter (HPF) using Projected Gradient Descent
% for a Least Squares problem with a unit simplex constraint.
% The model is given by:
% $$ 0.5 * || A * x - y ||_2^2     $$
% $$ subject to sum(x) = 1, x >= 0 $$
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

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;

DIFF_MODE_FORWARD   = 1;
DIFF_MODE_BACKWARD  = 2;
DIFF_MODE_CENTRAL   = 3;
DIFF_MODE_COMPLEX   = 4;

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;


%% Parameters

% Data
numCoeff    = 11;
numSamples  = 110;
noiseStd    = 0.075; %<! Try higher values to compare to LS
convShape   = CONVOLUTION_SHAPE_VALID;

% Numerical Differntiation
diffMode    = DIFF_MODE_CENTRAL;
errTol      = 1e-6;

% Solver
stepSizeMode    = STEP_SIZE_MODE_ADAPTIVE;
stepSize        = 0.001;
numIterations   = 2500;

% Visualization



%% Generate / Load Data
% The data model is a stream of samples going through an LTI system.  
% In our case the LTI system is built by an HPF filter:
% vY = conv(vX, vH) + vN
% where vH are the filter coefficients to estimate, vX is the data samples,
% vN is the AWGN and vY are the givem measurements.  
% In matrix form:
% vY = mX * vH + vN
% Here, mX is the convolution matrix built by teh data samples.

vX = sawtooth(1:numSamples);
for ii = 2:10
    vX = vX + sawtooth(ii:(numSamples + ii - 1));
end
vX = vX(:);

% Low Pass - Sum of 1 to keep DC, non negative.
vHRef = rand(numCoeff, 1);
vHRef = vHRef / sum(vHRef);

numSamplesY = numSamples - numCoeff + 1; %<! Output of the output ov `valid` convolution

vN = noiseStd * randn(numSamplesY, 1);
vY = conv(vX, vHRef, 'valid') + vN;

hObjFun = @(vH) 0.5 * sum((conv(vX, vH, 'valid') - vY) .^ 2);

% Analysis
mH = zeros(numCoeff, numIterations); %<! Initialization is the zero vector
vObjVal = zeros(numIterations, 1);


%% Display Data

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
set(hA, 'NextPlot', 'add');
hLineObj = plot(vY, 'DisplayName', 'Samples');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hA, 'Title'), 'String', {['Measured Data']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Value']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Samples Index']}, 'FontSize', fontSizeAxis);
ClickableLegend();


%% Convolution Matrix
%This section transforms the data `vX` into a convolution matrix `mX` such
%that `mX * vHRef ≈ vY`.
%
% Note: Farther down the we'll be able to solve the problem without
% matrices for better efficiency for large scale problem.

mX = CreateConvMtx1D(vX, numCoeff, convShape);

assertCond = norm(mX * vHRef - conv(vX, vHRef, 'valid'), 'inf') <= errTol;
assert(assertCond, 'The matrix convolution deviation exceeds the threshold %f', errTol);
disp(['The matrix convolution implementation is verified']);


%% Gradient Function
% 1. Derive the gradient of the objective function.
% 2. Implement it in `hObjFunGrad()`.

%----------------------------<Fill This>----------------------------%
hGradFun = @(vH) mX.' * (mX * vH - vY);
%-------------------------------------------------------------------%

vT = randn(numCoeff, 1);

vG = CalcFunGrad(vT, hObjFun, diffMode);
assertCond = norm(hGradFun(vT) - vG, 'inf') <= (errTol * norm(vG));
assert(assertCond, 'The gradient calculation deviation exceeds the threshold %f', errTol);

disp(['The gradient implementation is verified']);


%% Projection Function
% 1. Derive the projection function onto the unit simplex.
%    Note that it doesn't have a closed form solution.
% 2. Implement is as `ProjectSimplexBall(vY, ballRadius, epsVal)`.
% 3. Wrap it it as `hProjFun(vY, paramLambda)`.
%    The role of `paramLambda` will be explained.

%----------------------------<Fill This>----------------------------%
hProjFun = @(vY, paramLambda) ProjectSimplexBall(vY, 1);
%-------------------------------------------------------------------%

% Verify with CVX
vZ = randn(numCoeff, 1);
vS = hProjFun(vZ);

cvx_begin('quiet')
    cvx_precision('best');
    variable vT(numCoeff)
    minimize(0.5 * sum_square(vT - vZ));
    % minimize(0.5 * sum_square(mX * vT - vY));
    subject to
      sum(vT) == 1;
      vT >= 0;
cvx_end

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp([' ']);

assertCond = norm(vS - vT, 'inf') <= (errTol * norm(vT));
assert(assertCond, 'The projection calculation deviation exceeds the threshold %f', errTol);

disp(['The projection implementation is verified']);


%% Projected Gradient Descent
% 1. Implement Projected Gradient Descent in `ProxGradientDescent()`.
%    Later in the course the name will be explained.

mH = ProxGradientDescent(mH, hGradFun, hProjFun, stepSize);


%% Analysis

% Reference Solution
objValRef   = hObjFun(vHRef);

for ii = 1:numIterations
    vObjVal(ii) = hObjFun(mH(:, ii));
end

vObjVal = 20 * log10(abs(vObjVal - objValRef) / max(abs(objValRef), sqrt(eps())));

% Least Squares Solution
vLsSol = mX \ vY;
lsObjVal = 20 * log10(abs(hObjFun(vLsSol) - objValRef) / max(abs(objValRef), sqrt(eps())));


%% Display Results


figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numIterations, vObjVal, 'DisplayName', 'Projected Gradient Descent');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = yline(lsObjVal, 'DisplayName', 'Least Squares Solution');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hAxes, 'Title'), 'String', {['Objective Function Convergence']}, 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Index']}, 'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Relative Error [dB]']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

%?%?%?
% Why do we have this dip in the graph and then up? Think about the reference.


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

