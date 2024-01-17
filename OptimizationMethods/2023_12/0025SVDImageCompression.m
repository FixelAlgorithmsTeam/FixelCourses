% Optimization Methods
% SVD & Linear Least Squares - SVD Rank Approximation
% This scripts shows how to use rank approximation for image compression.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     12/01/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;


%% Parameters

% Data
imgFileName = 'Pullout.png';
paramK      = 8; %<! Working on patches with size (paramK, paramK)

% Model
% Set 9 different values
vSR = [3; 5; 10; 20; 30; 40; 50; 60; 64]; %<! Number of singular values for reconstruction

% Visualization



%% Generate / Load Data

mI = imread(imgFileName);
mI = mean(im2double(mI), 3); %<! Grayscale image

%?%?%?
% - Given the image dimensions and `paramK`. If each `paramK x paramK` patch
% is a column in a matrix, what would be the matrux dimensions?

mD = im2col(mI, [paramK, paramK], 'distinct'); %<! Each column is a block

numRows = size(mI, 1);
numCols = size(mI, 2);

tR = zeros(numRows, numCols, length(vSR)); %<! Reconstructed images


%% Display Data

figureIdx = figureIdx + 1;

hF = figure();
hA = axes();
imshow(mI);
set(get(hA, 'Title'), 'String', {['Input Image']}, 'FontSize', fontSizeTitle);

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


%% SVD and Singular Value Distribution
% 1. Calculate the mean patch from the data.
% 2. Make the data with zero mean.
% 3. Calculate the SVD of the centered data.

%----------------------------<Fill This>----------------------------%
vMeanD  = mean(mD, 2);
mD      = mD - vMeanD;
[mU, mS, mV] = svd(mD);
%-------------------------------------------------------------------%

% Analysis of the distribution of the singular values
vS          = diag(mS);
numSingVal  = length(vS);
vSEleEnergy = vS / sum(vS);
vSAccEnergy = cumsum(vS) / sum(vS);

figureIdx = figureIdx + 1;

hF = figure();
hA = axes(hF, 'NextPlot', 'add');
hSctObj = scatter(1:numSingVal, vSEleEnergy, 'filled', 'DisplayName', ['Normalized Energy']);
% hLineObj = plot(1:numSingVal, vSEleEnergy, 'DisplayName', ['Normalized Energy']);
% set(hLineObj, 'LineWidth', lineWidthNormal);
hLineObj = plot(1:numSingVal, vSAccEnergy, 'DisplayName', ['Accumulated Energy']);
set(hLineObj, 'LineWidth', lineWidthNormal);
set(get(hA, 'Title'), 'String', {['Singular Values Distribution']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Singular Value Index']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Normalized Value']}, 'FontSize', fontSizeAxis);
ClickableLegend('Location', 'northwest');

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


%% Projection onto the Columns Space

mRecPatches = zeros(size(mD));

for ii = 1:length(vSR)
    recRank = vSR(ii);
    mUd = mU(:, 1:recRank);
    mRecPatches(:) = mUd * (mUd.' * mD) + vMeanD;
    tR(:, :, ii) = col2im(mRecPatches, [paramK, paramK], [numRows, numCols], 'distinct');
end


%% Display Analysis
% Assumes the images are 320 x 320

figureIdx = figureIdx + 1;

hF = figure('Position', [50, 50, 1060, 1060]);
hA7 = axes(hF, 'Units', 'pixels', 'Position', [040, 010, 320, 320]);
hA8 = axes(hF, 'Units', 'pixels', 'Position', [380, 010, 320, 320]);
hA9 = axes(hF, 'Units', 'pixels', 'Position', [720, 010, 320, 320]);
hA4 = axes(hF, 'Units', 'pixels', 'Position', [040, 365, 320, 320]);
hA5 = axes(hF, 'Units', 'pixels', 'Position', [380, 365, 320, 320]);
hA6 = axes(hF, 'Units', 'pixels', 'Position', [720, 365, 320, 320]);
hA1 = axes(hF, 'Units', 'pixels', 'Position', [040, 715, 320, 320]);
hA2 = axes(hF, 'Units', 'pixels', 'Position', [380, 715, 320, 320]);
hA3 = axes(hF, 'Units', 'pixels', 'Position', [720, 715, 320, 320]);

vHa = [hA1; hA2; hA3; hA4; hA5; hA6; hA7; hA8; hA9];

for ii = 1:length(vHa)
    hImgObj = image(vHa(ii), repmat(tR(:, :, ii), 1, 1, 3));
    set(get(vHa(ii), 'Title'), 'String', {[num2str(vSR(ii)), ' singular values out of ', num2str(paramK * paramK)]}, 'FontSize', fontSizeTitle);
    set(vHa(ii), 'XTick', [], 'XTickLabel', []);
    set(vHa(ii), 'YTick', [], 'YTickLabel', []);
    set(get(vHa(ii), 'XAxis'), 'Visible', 'off');
    set(get(vHa(ii), 'YAxis'), 'Visible', 'off');
end

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


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

