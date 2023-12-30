% FISTA Accelerationn - Solving Linear Least Squares
% Compares FISTA Accelerated Gradient Descent vs. vanilla Gradient Descent.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     28/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;


%% Parameters

% Data
numRows = 100;
numCols = 30;

% Solver
stepSizeMode    = STEP_SIZE_MODE_CONSTANT;
stepSize        = 0.00035;
numIterations   = 5000;

% Visualization



%% Generate / Load Data

mA = randn(numRows, numCols);
vB = randn(numRows, 1);

hObjFun     = @(vX) 0.5 * sum((mA * vX - vB) .^ 2);
hGradFun    = @(vX) mA.' * (mA * vX - vB);
hProxFun    = @(vY, paramLambda) vY; %<! Identity -> Gradient Descent

mX = zeros(numCols, numIterations);

solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);


%% Solution by CVX

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default, Keep numRows low
% cvx_solver('SeDuMi');
% cvx_solver('Mosek'); %<! Can handle numRows > 500, Very Good!
% cvx_solver('Gurobi');

hRunTime = tic();

cvx_begin('quiet')
% cvx_begin()
    % cvx_precision('best');
    variable vX(numCols, 1);
    minimize( 0.5 * sum_square(mA * vX - vB));
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by GD'];

hRunTime = tic();

mX = GradientDescent(mX, hGradFun, stepSizeMode, stepSize);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX(:, end), runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by Accelerated Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by Accelerated GD'];

hRunTime = tic();

mX = ProxGradientDescentAccel(mX, hGradFun, hProxFun, stepSize);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX(:, end), runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Display Data

figureIdx = figureIdx + 1;
hF = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);

% hF = figure('Position', figPosLarge);
% hA = axes(hF, 'Units', 'pixels');
% set(hA, 'NextPlot', 'add');
% for ii = 1:length(vParamLambda)
%     paramLambda = vParamLambda(ii);
%     hLineObj = line(vX, hL1Prox(vX, paramLambda), 'DisplayName', ['\lambda = ', num2str(paramLambda, '%0.2f')]);
%     set(hLineObj, 'LineWidth', lineWidthNormal, 'Color', mColorOrder(ii, :));
% end
% % xline(0, 'HandleVisibility', 'off');
% % yline(0, 'HandleVisibility', 'off');
% set(hA, 'XGrid', 'on', 'XMinorGrid', 'off');
% set(hA, 'YGrid', 'on', 'YMinorGrid', 'off');
% set(get(hA, 'Title'), 'String', {['1D Shrinkage Operator']}, 'FontSize', fontSizeTitle);
% set(get(hA, 'XLabel'), 'String', {['Input Value']}, 'FontSize', fontSizeAxis);
% set(get(hA, 'YLabel'), 'String', {['Output Index']}, 'FontSize', fontSizeAxis);
% hLegend = ClickableLegend();
% set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

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
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
    % print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.svg'], '-vector', '-dsvg');
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

