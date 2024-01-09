% Optimization Methods
% SVD & Linear Least Squares - SVD Pseudo Inverse
% This scripts reproduce the examples in the slides.
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

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;


%% Example I

mA = [8, 10, 3, 30; 9, 6, 6, 18; 1, 1, 10, 3];
vX = [1; 2; 3; 6];
vB = mA * vX;

% What is the rank of A?
%----------------------------<Fill This>----------------------------%
% Use the SVD to calculate the rank of the matrix
[mU, mS, mV] = svd(mA);
rankA = sum(diag(mS) ~= 0);
%-------------------------------------------------------------------%

%?%?%?
% - What does it mean about A?
% - Is mA.' * mA SPD or SPSD? Why? You may calculate to see.

% Calculate the pseudo inverse of mS.
%----------------------------<Fill This>----------------------------%
% Use the SVD to calculate the rank of the matrix
mSI = mS.';
for ii = 1:min(size(mSI))
    mSI(ii, ii) = ((mSI(ii, ii) ~=0) + 0) / (mSI(ii, ii) + (mSI(ii, ii) == 0) * 1);
end
%-------------------------------------------------------------------%

% Calculate the SVD based Pseudo Inverse of A
%----------------------------<Fill This>----------------------------%
% Use the SVD to calculate the pseudo inverse of the matrix
mAPInv = mV * mSI * mU.';
%-------------------------------------------------------------------%

% Solve the equation using the Pseudo Inverse
%----------------------------<Fill This>----------------------------%
vXEst = mAPInv * vB;
%-------------------------------------------------------------------%

%?%?%?
% - Will the solution to `mA \ vB` be any different?


%% Example II

mA = [5, 0, 0, 0; 0, 2, 0, 0; 0, 0, 0, 0];
vB = [5; 4; 3];

% What is the rank of A?
%----------------------------<Fill This>----------------------------%
% Use MATLAB's build in function to calculate the rank.
rankA = rank(mA);
%-------------------------------------------------------------------%

[mU, mS, mV] = svd(mA);

%?%?%?
% - What is the null space of A?

% Calculate the pseudo inverse of mS.
%----------------------------<Fill This>----------------------------%
% Use the SVD to calculate the rank of the matrix
mSI = mS.';
for ii = 1:min(size(mSI))
    mSI(ii, ii) = ((mSI(ii, ii) ~=0) + 0) / (mSI(ii, ii) + (mSI(ii, ii) == 0) * 1);
end
%-------------------------------------------------------------------%

% Calculate the SVD based Pseudo Inverse of A
%----------------------------<Fill This>----------------------------%
% Use the SVD to calculate the pseudo inverse of the matrix
mAPInv = mV * mSI * mU.';
%-------------------------------------------------------------------%

% Solve the equation using the Pseudo Inverse
%----------------------------<Fill This>----------------------------%
vXEst = mAPInv * vB;
%-------------------------------------------------------------------%

%?%?%?
% - Will the solution to `mA \ vB` be any different?
% - Does the solution solve the linear system?

% Calculate b̂ = P_R(A) (b) = sum_i^r {u}_{i}^{T} b {u}_{i}
%----------------------------<Fill This>----------------------------%
vBHat = mU(:, 1:2) * mU(:, 1:2).' * vB; %<! Matches the loop
%-------------------------------------------------------------------%

% Compare the following solution:
% 1. LS using MATLAB's `\`.
% 2. LS Using SVD Pseudo Inverse.
%----------------------------<Fill This>----------------------------%
mA \ vB
pinv(mA) * vB
%-------------------------------------------------------------------%

%?%?%?
% - Do they solve the equation `mA * vX = vB`?
% - Do they solve the equation `mA * vX = vBHat`?

mA(2, 4) = 4;
mA \ vB
pinv(mA) * vB

% - Why are the solution above different?


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

