% Optimization Methods
% Convex Optimization - Algorithms & Solvers - ADMM for TV Deblurring
% Using ADMM to solve:
% $$ arg min_x 0.5 * || H x - y ||_2^2 + λ || D * x ||_1 $$
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

generateFigures = OFF;

%% Constants

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;

CONVOLUTION_SHAPE_FULL  = 1;
CONVOLUTION_SHAPE_SAME  = 2;
CONVOLUTION_SHAPE_VALID = 3;


%% Parameters

% Data
numSamples  = 250;
noiseStd    = 0.075;

filterRadius = 7;

% Model
convType    = CONVOLUTION_SHAPE_VALID;
paramLambda = 0.075;

% Solver
stepSize        = 0.015;
paramRho        = 2;
numIterations   = 10000;

% Visualization



%% Generate / Load Data

%?%?%?
% - What is the dimensions of vX by the model?

numSamplesFilter = 2 * filterRadius + 1;
numSamplesX = numSamples + numSamplesFilter - 1;

% Filter
vH = ones(numSamplesFilter, 1) / numSamplesFilter;

% Data
vS = MakeSignal('Blocks', numSamplesX);
vS = vS(:);
vZ = vS + (noiseStd * randn(size(vS, 1), 1));
vY = conv2(vZ, vH, 'valid'); %<! `conv()` is a wrapper of `conv2()`

% Model
mD = spdiags([-ones(numSamplesX - 1, 1), ones(numSamplesX - 1, 1)], [0, 1], numSamplesX - 1, numSamplesX);
mH = CreateConvMtx1D(vH, numSamplesX, convType);

% Solvers
mX = zeros(numSamplesX, numIterations);

solverIdx       = 0;
cMethodString   = {};

mObjFunValMse   = zeros([numIterations, 1]);
mSolMse         = zeros([numIterations, 1]);

hObjFun = @(vX) 0.5 * sum((conv(vX, vH, 'valid') - vY) .^ 2) + paramLambda * norm(mD * vX, 1);


%% Display the Data

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
set(hA, 'NextPlot', 'add');
hLineObj = line(1:size(vS, 1), vS, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = line(1:size(vZ, 1), vZ, 'DisplayName', 'Data Samples');
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');

set(get(hA, 'Title'), 'String', {['Model Data and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['x']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['y']}, 'FontSize', fontSizeAxis, 'Interpreter', 'latex');

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end


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
    variable vX(numSamplesX);
    minimize( 0.5 * sum_square(mH * vX - vY) + paramLambda * norm(mD * vX, 1));
%-------------------------------------------------------------------%
cvx_end

runTime = toc(hRunTime);

DisplayRunSummary(solverString, hObjFun, vX, runTime, cvx_status);

sCvxSol.vXCvx     = vX;
sCvxSol.cvxOptVal = hObjFun(vX);


%% Solution by Accelerated Gradient Descent
% 1. Calculate the Sub Gradient step.

%----------------------------<Fill This>----------------------------%
hGradFun = @(vX) mH.' * (conv(vX, vH, 'valid') - vY) + paramLambda * mD.' * sign(mD * vX);
%-------------------------------------------------------------------%

%?%?%?
% - How could it be implemented without a matrix multiplication? Think operators.

solverIdx                   = solverIdx + 1;
cLegendString{solverIdx}    = ['Solution by Accelerated Sub Gradient'];

hRunTime = tic();

mX = ProxGradientDescentAccel(mX, hGradFun, @(vY, paramLambda) vY, stepSize, paramLambda);

runTime = toc(hRunTime);

DisplayRunSummary(cLegendString{solverIdx}, hObjFun, mX(:, end), runTime);
[mObjFunValMse, mSolMse] = UpdateAnalysisData(mObjFunValMse, mSolMse, mX, hObjFun, sCvxSol, solverIdx);


%% Solution by ADMM
% 1. Set `hMinFun = @(vZ, vW, paramRho) ...`.  
%    It minimizes the term with regard to x.
%    You may assume `paramRho` is constant.
% 2. Set `hProxFun = @(vY, paramLambda) ...`.
%    It applies the Proximal Operator with reagrd to g().

% You may find this useful
mDD = paramRho * (mD.' * mD); %<! In practice, much better use operators
mHH = mH.' * mH;
vHy = mH.' * vY;
mHDC = decomposition(mHH + mDD, 'chol'); %!< Assuming their null space is exclusive

%----------------------------<Fill This>----------------------------%
hMinFun     = @(vZ, vW, paramRho) mHDC \ (vHy + paramRho * mD.' * (vZ - vW));
hProxFun    = @(vY, paramLambda) max(abs(vY) - paramLambda, 0) .* sign(vY); %<! Prox L1
%-------------------------------------------------------------------%

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
hLineObj = line(1:size(vS, 1), vS, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = line((filterRadius + 1):(size(vY, 1) + filterRadius), vY, 'DisplayName', 'Data Samples');
set(hLineObj, 'LineStyle', 'none', 'Marker', '*');
hLineObj = line(1:size(mX, 1), mX(:, end), 'DisplayName', 'TV Deblurring');
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

