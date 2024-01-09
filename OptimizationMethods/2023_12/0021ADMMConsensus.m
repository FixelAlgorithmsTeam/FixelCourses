% Optimization Methods
% Convex Optimization - Algorithms & Solvers - Consensus Trick
% Using ADMM to solve:
% $$ arg min_x 0.5 * || x - y ||_2^2 subject to sum(x) = 1, x >= 0 $$
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     30/12/2023
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
numSamples  = 200;

% Solver
stepSize        = 0.0025;
paramRho        = 2;
numIterations   = 2500;

% Visualization



%% Generate / Load Data

vS = MakeSignal('Blocks', numSamples);
vS = vS(:);
vY = vS + (noiseStd * randn(numSamples, 1));
mD = spdiags([-ones(numSamples - 1, 1), ones(numSamples - 1, 1)], [0, 1], numSamples - 1, numSamples);

% Solvers
mX = zeros(numSamples, numIterations);

solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);

hObjFun = @(vX) 0.5 * sum((vX - vY) .^ 2) + paramLambda * norm(mD * vX, 1);


%% Display the Data

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
set(hA, 'NextPlot', 'add');
hLineObj = line(1:numSamples, vS, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = line(1:numSamples, vY, 'DisplayName', 'Data Samples');
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');

set(get(hA, 'Title'), 'String', {['Model Data and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['y']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

%?%?%?
% - How would the least squares (With no regularization) solution look like?


%% Solution by DCP (CVX)
% 1. Formulate the TV Denoising problem as a DCP problem.
% 2. Solve the problem using DCP (CVX). 
%    Save the output as `vX`.

solverString = 'CVX';

% cvx_solver('SDPT3'); %<! Default
% cvx_solver('SeDuMi');

hRunTime = tic();

cvx_begin('quiet')
%----------------------------<Fill This>----------------------------%
    variable vX(numSamples, 1);
    minimize( 0.5 * sum_square(vX - vY) + paramLambda * norm(mD * vX, 1));
%-------------------------------------------------------------------%
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Implement ADMM Function
% This section implements the ADMM function and its auxiliary function.
% 1. Implement `ADMM( mX, hMinFun, hProxFun, mP, paramRho, paramLambda, vZ, vW )`.
% 2. Set `hMinFun = @(vZ, vW, paramRho) ...`.  
%    It minizes the term with regard to x.
%    You may assume `paramRho` is constant.
% 3. Set `hProxFun = @(vY, paramLambda) ...`.
%    It applies the Proximal Operator with reagrd to g().

% You may find this useful
mDD = speye(numSamples) + paramRho * (mD.' * mD); %<! In practive, much better use operators
mDC = decomposition(mDD, 'chol');

%----------------------------<Fill This>----------------------------%
hMinFun     = @(vZ, vW, paramRho) mDC \ (vY + paramRho * mD.' * (vZ - vW));
hProxFun    = @(vY, paramLambda) max(abs(vY) - paramLambda, 0) .* sign(vY); %<! Prox L1
%-------------------------------------------------------------------%

% For the FISTA Sub Gradient
hGradFun = @(vX) (vX - vY) + paramLambda * mD.' * sign(mD * vX);


%% Solution by Accelerated Gradient Descent

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by Accelerated Sub Gradient'];

hRunTime = tic();

mX = ProxGradientDescentAccel(mX, hGradFun, @(vY, paramLambda) vY, stepSize, paramLambda);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX(:, end), runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by ADMM

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by ADMM'];

hRunTime = tic();

mX = ADMM(mX, hMinFun, hProxFun, mD, paramRho, paramLambda);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX(:, end), runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Display Data

figureIdx = figureIdx + 1;
hF = DisplayComparisonSummary(numIterations, mObjFunValMse, mSolMse, cLegendString, figPosLarge, lineWidthNormal, fontSizeTitle, fontSizeAxis);

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
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
set(hA, 'NextPlot', 'add');
hLineObj = line(1:numSamples, vS, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = line(1:numSamples, vY, 'DisplayName', 'Data Samples');
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');
hLineObj = line(1:numSamples, mX(:, end), 'DisplayName', 'TV Denoising');
set(hLineObj, 'LineWidth', lineWidthNormal);

set(get(hA, 'Title'), 'String', {['Model Data and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['y']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

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
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end

%?%?%?
% - How can we imporve the results?
%   Think of the step size of the PGD vs. what's needed for Sub Gradient method.


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

