% Optimization Methods
% Convex Optimization - Constraint Optimization - DCP
% Calculating the minimum area circle.
% The model is given by:
% $$ \arg \min_{c, r} r $$
% $$ subject to || xi - c ||_2 <= r
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
numSamples  = 20;
dataDim     = 2;

% Visualization
vLim = [-2; 2];


%% Generate / Load Data

mX = 2 * (rand(dataDim, numSamples) - 0.5);


%% Display Data

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
set(hA, 'NextPlot', 'add');
mColorOrder = get(hA, 'ColorOrder');

hSctrObj = scatter(mX(1, :), mX(2, :), 'filled', 'DisplayName', 'Data');
set(hSctrObj, 'SizeData', 75);


set(hA, 'XLim',vLim, 'YLim', vLim);
set(hA, 'DataAspectRatio', [1,1 ,1]);
set(get(hA, 'XLabel'), 'String', {['x_1']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['x_2']}, 'FontSize', fontSizeAxis);



%% DCP Optimization
% 1. Formulate the problem in CVX.
%    Use `valRadius` for the radius and `vC` for the center.

cvx_solver('SDPT3'); %<! Default
% cvx_solver('SeDuMi'); %<! Faster than 'SDPT3', yet less accurate

hRunTime = tic();

cvx_begin('quiet')
    % cvx_precision('best');
    variables valRadius vC(dataDim)
    minimize(valRadius);
    subject to
      for ii = 1:numSamples
          norm(mX(:, ii) - vC) <= valRadius;
      end
cvx_end

runTime = toc(hRunTime);

disp([' ']);
disp(['CVX Solution Summary']);
disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Run Time Is Given By - ', num2str(runTime), ' [Sec]']);
disp([' ']);


%% Display Results

set(get(hA, 'Title'), 'String', {['Minimum Area Bounding Circle by DCP']}, 'FontSize', fontSizeTitle);
hRectObj = rectangle('Position', [vC(1) - valRadius, vC(2) - valRadius, 2 * valRadius, 2 * valRadius], 'Curvature', [1, 1]);
set(hRectObj, 'EdgeColor', 'red', 'LineWidth', lineWidthNormal);
hSctrObj = scatter(vC(1), vC(2), 'filled', 'DisplayName', 'Bounding Circle');
set(hSctrObj, 'SizeData', 100, 'MarkerFaceColor', 'r');
ClickableLegend();



%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

