% Optimization Methods
% SVD & Linear Least Squares - Solving Multiple LS with the Same Model.
% Solving:
% $$ \arg \min_{x} 0.5 * || A * x - b_i ||_2^2 i = 1, 2, ...
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

%% Constants


%% Parameters

numRows = 500;
numCols = 100;
numIn   = 1000; %<! Number of Inputs

% Visualization



%% Generate / Load Data

mA = randn(numRows, numCols);
mB = randn(numRows, numIn);

% Refernce Solution
mXRef = mA \ mB;

mX = zeros(numCols, numIn);


%% Solution by Pre Calculation

sOpts = struct();

mC = mA.' * mA;
sOpts.SYM       = true();
sOpts.POSDEF    = true();

hRunTime = tic();

for ii = 1:numIn
    mX(:, ii) = linsolve(mC, mA.' * mB(:, ii), sOpts);
end

runTime = toc(hRunTime);

disp(['Total run time: ', num2str(runTime), ' [Sec].']);


%% Solution by Decomposition

mC = decomposition(mA);

hRunTime = tic();

for ii = 1:numIn
    mX(:, ii) = mC \ mB(:, ii);
end

runTime = toc(hRunTime);

disp(['Total run time: ', num2str(runTime), ' [Sec].']);

%?%?%?
% - What if A was symmetric? How can we take advantage of that?
% - What if A was SPD? How can we take advantage of that?
% - What if A was orthonormal?


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

