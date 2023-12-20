% Optimization Methods
% Convex Optimization - Non Smooth Optimization - Sub Gradient Method
% Minimizing the maximum of a set of a functions.
% The model is given by:
% $$ \arg \min_{x} \max_{i} a_i^x + b $$
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     13/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;
% subStreamNumberDefault = 0;

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
numRows = 30; %<! Num functions
numCols = 5; %<! Data Dimensions

% Solver
numIterations = 10000;

% Visualization
vLim = [-2; 2];


%% Generate / Load Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);

% Analysis
vObjVal = zeros(numIterations, 1);

hObjFun = @(vX) max(mA * vX + vB);


%% Sub Gradient Method
% 1. Derive the subgradient of the function.
% 2. Immplement the Sub Gradient as `hSubGradFun(vX)`.
% 2. Implement the Sub Gradient Method. Choose the step size correctly.

hSubGradFun = @(vX) mean(mA((mA * vX + vB) == max(mA * vX + vB), :), 1).';

vX = zeros(numCols, 1);
vG = zeros(numCols, 1);

vObjVal(1) = hObjFun(vX);

for ii = 2:numIterations
    vG(:) = hSubGradFun(vX);
    stepSize = 1 / ii;
    vX(:) = vX - stepSize * vG;
    vObjVal(ii) = hObjFun(vX);
end


%% DCP Optimization
% 1. Formulate the problem in CVX.
%    Use vXRef for the optimal argument.

% cvx_solver('SDPT3'); %<! Default
cvx_solver('SeDuMi'); %<! Faster than 'SDPT3', yet less accurate

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variables vXRef(numCols) t
    minimize(max(mA * vXRef + vB));
    % minimize(t);
    % subject to
    %     max(mA * vXRef + vB) <= t;
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Analysis

objValRef   = hObjFun(vXRef);
vObjVal = 20 * log10(abs(vObjVal - objValRef) / max(abs(objValRef), sqrt(eps())));


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', figPosLarge);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
hLineObj = plot(1:numIterations, vObjVal, 'DisplayName', 'Sub Gradient Method');
set(hLineObj, 'LineWidth', lineWidthNormal);

set(get(hAxes, 'Title'), 'String', {['Objective Function Convergence']}, 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Index']}, 'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Relative Error [dB]']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

%?%?%?
% Why is the objective not monotonic decreasing?
% What will happen if we set numRows ~= numCols? Think about the random directions.



%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

