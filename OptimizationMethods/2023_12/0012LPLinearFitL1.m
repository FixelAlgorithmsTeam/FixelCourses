% Optimization Methods
% Convex Optimization - Constraint Optimization - Linear Fit with L1 Norm
% Using Linear Programming formulation to find the optimal linear fit with
% regard to L1 norm.
% The model is given by:
% $$ || A * x - y ||_1 $$
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
numSamples  = 25;
numOutliers = 5;
noiseStd    = 0.05;
outlierStd  = 1.5;

% Model ax + b
modelSlope      = 1;
modelIntercept  = 0;

% Verification
errTol      = 1e-6;

% Visualization


%% Generate / Load Data

vG = rand(numSamples + numOutliers, 1);
vY = ((modelSlope * vG) + modelIntercept) + (noiseStd * randn(numSamples + numOutliers, 1));
vI = randperm(numSamples + numOutliers, numOutliers);
vY(vI) = vY(vI) + (outlierStd * randn(numOutliers, 1));

mA = cat(2, ones(length(vG), 1), vG);


%% Linear Least Squares Solution
% 1. Solve the problem using Least Squares.
% 2. Save the result as `vXLs`.

%----------------------------<Fill This>----------------------------%
vXLs = mA \ vY;
%-------------------------------------------------------------------%


%% Display Data

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
set(hA, 'NextPlot', 'add');
hLineObj = plot(vG, ((modelSlope * vG) + modelIntercept), 'DisplayName', 'Model');
hSctrObj = scatter(vG, vY, 'filled', 'DisplayName', 'Samples');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(vG, ((vXLs(2) * vG) + vXLs(1)), 'DisplayName', 'Least Squares Model');
set(hLineObj, 'LineWidth', lineWidthNormal);
set(hSctrObj, 'SizeData', 50);
set(get(hA, 'Title'), 'String', {['Model & Measured Data']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['y']}, 'FontSize', fontSizeAxis);
ClickableLegend();


%% Linear Programming Solution
% 1. Formulate the L1 problem as a Linear Programming problem.
% 2. Solve the problem using DCP (CVX). 
%    Save the output as `vXL1`.

hRunTime = tic();

cvx_begin('quiet')
%----------------------------<Fill This>----------------------------%
    variables vT(numSamples + numOutliers) vXL1(2)
    minimize( sum(vT) );
    subject to
      mA * vXL1 - vY <= vT;
      vY - mA * vXL1 <= vT;
%-------------------------------------------------------------------%      
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Direct Solution
% Solving the model using DCP.
% 1. Formulate the L1 problem using DCP (CVX).

hRunTime = tic();

cvx_begin('quiet')
%----------------------------<Fill This>----------------------------%
    variables vXL1Ref(2)
    minimize( norm(mA * vXL1Ref - vY, 1) );
%-------------------------------------------------------------------%  
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);

assertCond = norm(vXL1Ref - vXL1, 'inf') <= errTol * norm(vXL1Ref);
assert(assertCond, 'The LP solution deviation exceeds the threshold %f', errTol);
disp(['The LP solution implementation is verified']);


%% Display Results


hLineObj = plot(vG, ((vXL1(2) * vG) + vXL1(1)), 'DisplayName', 'L1 Model');
set(hLineObj, 'LineWidth', lineWidthNormal);

%?%?%?
% Which formulation would you use in large scale scenario?

%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

