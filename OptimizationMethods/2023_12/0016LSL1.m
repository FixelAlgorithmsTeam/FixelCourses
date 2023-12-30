% Optimization Methods
% Convex Optimization - Non Smooth Optimization - Proximal Gradient Method
% Using the LASSO model for selection of features.
% The model is given by:
% $$ arg min_x || A * x - y ||_2^2 + λ || x ||_1 $$
% Since the objective promotes sparsity, we can use it for feature
% selection.
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
csvUrl = 'https://raw.githubusercontent.com/FixelAlgorithmsTeam/FixelCourses/master/DataSets/mtcars.csv';

% Solution Path
vParamLambda = linspace(0, 7, 100);

% Solver
stepSize        = 0.0005;
numIterations   = 50000;

% Verification
diffMode    = DIFF_MODE_CENTRAL;
errTol      = 1e-5;

% Visualization


%% Generate / Load Data

% Data from https://gist.github.com/seankross/a412dfbd88b3db70b74b
% mpg - Miles per Gallon
% cyl - # of cylinders
% disp - displacement, in cubic inches
% hp - horsepower
% drat - driveshaft ratio
% wt - weight
% qsec - 1/4 mile time; a measure of acceleration
% vs - 'V' or straight - engine shape
% am - transmission; auto or manual
% gear - # of gears
% carb - # of carburetors
taData = readtable(csvUrl);
mA = table2array(taData(:, 3:end)); %<! Removing model and mpg
mA = (mA - mean(mA)) ./ std(mA); %<! Normalize
vY = taData{:, 2}; %<! Extracting mpg

cFeatureName = taData.Properties.VariableNames(3:end);

numSamples      = size(vY, 1);
numFeatures     = size(mA, 2);
numParamLambda  = length(vParamLambda);

mX          = zeros(numFeatures, numIterations);
mXLambda    = zeros(numFeatures, numParamLambda);

hObjFun = @(vX, paramLambda) 0.5 * sum((mA * vX - vY) .^ 2) + paramLambda * norm(vX, 1);


%% Proximal Gradient Descent
% 1. Create a function called `ProxGradientDescent(mX, hGradFun, hProxFun, stepSize, paramLambda)`.
% 2. Implement the proximal gradient descent in the function.
% 3. Run this section to verify your implementation.

hGradFun = @(vX) vX - vY(1:numFeatures);
hProxFun = @(vY, paramLambda) max(0, 1 - (paramLambda / norm(vY))) * vY; %<! L2 Prox

mX(:, 1) = vY(1:numFeatures);
mX = ProxGradientDescent(mX, hGradFun, hProxFun, stepSize, vParamLambda(end));

cvx_begin('quiet')
    cvx_precision('best');
    variables vXRef(numFeatures)
    minimize(0.5 * sum_square(vXRef - vY(1:numFeatures)) + vParamLambda(end) * norm(vXRef));
cvx_end

assertCond = norm(mX(:, end) - vXRef, 'inf') <= errTol;
assert(assertCond, 'The PGD solution deviation exceeds the threshold %f', errTol);
disp(['The PGD implementation is verified']);


%% Set Auxiliary Functions
% 1. Set `hGradFun = @(vX) ...` to calculate the gradient of f(x).
% 2. Set `hProxFun = @(vY, paramLambda) ...` to calculate the proximal operator of g(x).
% 3. Run this section to verify your implementation.

%----------------------------<Fill This>----------------------------%
hGradFun = @(vX) mA.' * (mA * vX - vY);
hProxFun = @(vY, paramLambda) max(abs(vY) - paramLambda, 0) .* sign(vY); %<! L1 Prox
%-------------------------------------------------------------------%

vX = randn(numFeatures, 1);

vG = CalcFunGrad(vX, @(vX) 0.5 * sum((mA * vX - vY) .^ 2), diffMode);
assertCond = norm(hGradFun(vX) - vG, 'inf') <= (errTol * norm(vG));
assert(assertCond, 'The Gradient Operator calculation deviation exceeds the threshold %f', errTol);

disp(['The Gradient Operator implementation is verified']);

cvx_begin('quiet')
    cvx_precision('best');
    variables vXRef(numSamples)
    minimize(0.5 * sum_square(vXRef - vY) + vParamLambda(end) * norm(vXRef, 1)); %<! Prox
cvx_end

assertCond = norm(hProxFun(vY, vParamLambda(end)) - vXRef, 'inf') <= errTol;
assert(assertCond, 'The Proximal Operator calculation deviation exceeds the threshold %f', errTol);
disp(['The Proximal Operator implementation is verified']);


%% Analysis

% Calculating the feature significance per λ.
for ii = 1:numParamLambda
    mX = ProxGradientDescent(mX, hGradFun, hProxFun, stepSize, numSamples * vParamLambda(ii));
    mXLambda(:, ii) = abs(mX(:, end)); %<! Significance is in absolute value
end


%% Display Results

figureIdx = figureIdx + 1;

hFigure = figure('Position', [50, 50, 1500, 700]);
hAxes   = axes(hFigure);
set(hAxes, 'NextPlot', 'add');
for ii = 1:numFeatures
    hLineObj = line(vParamLambda, mXLambda(ii, :), 'DisplayName', cFeatureName{ii});
    set(hLineObj, 'LineWidth', lineWidthNormal);
end
set(get(hAxes, 'Title'), 'String', {['Feature Significance to Estimate MPG']}, 'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['λ']}, 'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Significance']}, 'FontSize', fontSizeAxis);

hLegend = ClickableLegend();

if(generateFigures == ON)
    print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.png'], '-dpng', '-r0'); %<! Saves as Screen Resolution
end

%?%?%?
% Why is the significance not monotonic?



%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

