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
% - 1.0.000     06/01/2024
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Parameters

% Data
imgFileName = 'Pullout.png';
paramK      = 8; %<! Working on patches with size (paramK, paramK)

% Model
vS = [1; 5; 10; 15; 20; 30; 40; 50; 60];

% Visualization



%% Generate / Load Data

mI = imread(imgFileName);
mI = mean(im2double(mI), 3); %<! Grayscale image

%?%?%?
% - Give the image dimensions and `paramK`. If each `paramK x paramK` patch
% is a column in a matrix, what would be the matrux dimensions?

mD = im2col(mI, [paramK, paramK], 'distinct'); %<! Each column is a block


%% Display Data

hF = figure();
hA = axes();
imshow(mI);
set(get(hA, 'Title'), 'String', {['Input Image']}, 'FontSize', fontSizeTitle);


%% Analysis

[mU, mS, mV] = svd(mD);

%?%?%?
% - Which matrix is needed for the compression, U or V?



%% Display Analysis


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

