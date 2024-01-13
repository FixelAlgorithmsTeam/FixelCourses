% Optimization Methods
% SVD & Linear Least Squares - Regularized Least Squares.
% Solving:
% $$ \arg \min_{x} 0.5 * || A * x - y ||_2^2 + λ ||x||_2^2
% This secript shows the ability of λ to assist with preventing overfit.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     13/01/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 33;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants


%% Parameters

% Data
polyDeg     = 6;
noiseStd    = 0.5;
numSamples  = 100;
gridMinVal  = 0;
gridMaxVal  = 1.5;

trainDataRatio = 0.15;

% Model
vLambda     = linspace(0, 50, 5000) / numSamples;
vPolyDeg    = 4:20;
vPolyDeg    = vPolyDeg(:);

% Visualization



%% Generate / Load Data

% The whole data
vARef = linspace(gridMinVal, gridMaxVal, numSamples);
vARef = vARef(:);
mARef = vARef .^ (0:polyDeg);

vX = 1 * randn(polyDeg + 1, 1);

vZ = mARef * vX;
vN = noiseStd * randn(numSamples, 1);
vY = vZ + vN;

mAModel = vARef .^ (0:max(vPolyDeg));

[vIdxTrain, vIdxTest] = PartitionTrainTest(numSamples, trainDataRatio);
mA      = mAModel(vIdxTrain, :);
vB      = vY(vIdxTrain);
mATest  = mAModel(vIdxTest, :);
vBTest  = vY(vIdxTest);


%% Display Data

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vARef, vZ, 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(vARef, vY, 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
set(get(hA, 'Title'), 'String', {['Model and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Value']}, 'FontSize', fontSizeAxis);
ClickableLegend();

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

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vARef(vIdxTrain), vZ(vIdxTrain), 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(vARef(vIdxTrain), vY(vIdxTrain), 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
set(get(hA, 'Title'), 'String', {['Train Model and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Value']}, 'FontSize', fontSizeAxis);
ClickableLegend();

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

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hLineObj = plot(vARef(vIdxTest), vZ(vIdxTest), 'DisplayName', 'Model Data');
set(hLineObj, 'LineWidth', lineWidthNormal);
scatter(vARef(vIdxTest), vY(vIdxTest), 20, 'filled', 'DisplayName', 'Data Samples'); %<! Color as noise level
set(get(hA, 'Title'), 'String', {['Test Model and Noisy Samples']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Sample Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Value']}, 'FontSize', fontSizeAxis);
ClickableLegend();

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


%% Polynomial Degree vs. Regularization Factor

mZTrain = zeros(length(vPolyDeg), length(vLambda));
mZTest  = zeros(length(vPolyDeg), length(vLambda));

for jj = 1:length(vLambda)
    paramLambda = vLambda(jj);
    for ii = 1:length(vPolyDeg)
        paramP  = vPolyDeg(ii);
        mAP     = mA(:, 1:(paramP + 1));
        if paramLambda == 0
            vXRls   = pinv(mAP.' * mAP + paramLambda * eye(paramP + 1)) * (mAP.' * vB);
        else
            vXRls   = (mAP.' * mAP + paramLambda * eye(paramP + 1)) \ (mAP.' * vB);
        end

        mZTrain(ii, jj) = sqrt(mean((mAP * vXRls - vB) .^ 2));
        mZTest(ii, jj) = sqrt(mean((mATest(:, 1:(paramP + 1)) * vXRls - vBTest) .^ 2));
    end
end

%?%?%?
% - How can the loop above be optimized?

% figure(); imagesc(vLambda, vPolyDeg, 20 * log10(1 + mZTest));
% % figure(); imagesc(vLambda, vPolyDeg, mZTest);
% title('Test');
% figure(); imagesc(vLambda, vPolyDeg, 20 * log10(1 + mZTrain));
% % figure(); imagesc(vLambda, vPolyDeg, mZTrain);
% title('Train');

% figure(); plot(vLambda, mZTest(2, :));
% figure(); plot(vLambda, mZTest(4, :));
% figure(); plot(vLambda, mZTest(6, :));
% figure(); plot(vLambda, mZTest(end, :));


%% Display Analysis

figureIdx = figureIdx + 1;

hF = figure('Position', figPosLarge);
hA = axes(hF);
hSurfObj = surf(vLambda, vPolyDeg, 20 * log10(1 + mZTrain));
set(hSurfObj, 'EdgeColor', 'none');
set(get(hA, 'Title'), 'String', {['Estimation RMSE - Train']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['λ']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Polynomial Degree']}, 'FontSize', fontSizeAxis);
set(get(hA, 'ZLabel'), 'String', {['RMSE [dB]']}, 'FontSize', fontSizeAxis);

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
hSurfObj = surf(vLambda, vPolyDeg, 20 * log10(1 + mZTest));
set(hSurfObj, 'EdgeColor', 'none');
set(get(hA, 'Title'), 'String', {['Estimation RMSE - Test']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['λ']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Polynomial Degree']}, 'FontSize', fontSizeAxis);
set(get(hA, 'ZLabel'), 'String', {['RMSE [dB]']}, 'FontSize', fontSizeAxis);

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


%% Auxiliary Functions

function [vTrainIdx, vTestIdx] = PartitionTrainTest(numSamples, trainRatio);

numTrainSamples = round(trainRatio * numSamples);
vTrainIdx       = sort(randperm(numSamples, numTrainSamples), 'ascend');
vTestIdx        = setdiff(1:numSamples, vTrainIdx);

end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

