% Optimization Methods
% Convex Optimization - Constraint Optimization - Solve Sudoku
% Solving a 9x9 Sudoku board using Linear Programming.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     19/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;

%% Constants

BOARD_NUM_ROWS =  9;
BOARD_NUM_COLS =  BOARD_NUM_ROWS;


%% Parameters

sudokuBoardFileName = 'SudokuBoard.mat';


%% Generate / Load Data

% Loads the Sudoku board (`mB`)
% mB(ii, jj, cellVal): ii - Row, jj - Col, cellVal - Value
load(sudokuBoardFileName);


%% Display Data

[hF, hA] = DrawSudokuBoard(mB, BOARD_NUM_ROWS);


%% Formulate Problem

numRows = BOARD_NUM_ROWS;

% For Integer Programming we could just create a 2D array and impose
% constarints and values. 
% Yet to use Linear Programming we will use a binary formulation by setting
% a 3D tensor `tX` where `[9, 9, 9] = size(tX)` and if `tX(ii, jj, kk) = 1`
% it suggests that the value of the `ii, jj` cell on the board is `kk`.

numVar = numRows ^ 3;

% Impose Constraints
% While conceptually the data is 3D tensor, in practice we solve:
% arg min_x    f^t * x    (LP Objective)
% subject to   A * x = b  (Equality Constraint)
%             0 <= x <= 1
% 1. Each column      `sum(tX(:, jj, kk)) = 1`.
% 2. Each row         `sum(tX(ii, :, kk)) = 1`.
% 3. Each depth slice `sum(tX(ii, jj, :)) = 1`.
% 4. Each sub grid    `sum(tX(..., ..., kk)) = 1`.
% 5. For each given index `tX(ii, jj, clueVal) = 1`.  
%    We can also limit the lower value for those indices to 1.
% 6. Continuous binary variable `0 <= tX <= 1`.

vF = zeros(numVar, 1);
numClues = size(mB, 1);
numConst = 4 * (numRows ^ 2); %<! Equality to clues using lower bounds

% Constarint Matrix
% mA * vX = vB;
% Assuming `vX = tX(:)` -> Column based
mA = zeros(numConst, numVar);

conA = 0; %<! Index of the constraint
% Columns Constraits
itmIdx = 1; %<! First item in Column / Row / 3rd Dim Slice index
for ii = 1:(numRows * numRows)
    conA = conA + 1;
    mA(conA, itmIdx:(itmIdx + numRows - 1)) = 1;
    itmIdx = itmIdx + numRows;
end

%?%?%?
% Can you do it using `kron()`?

% Rows Constraits
itmIdx = 1; %<! First item in Column / Row / 3rd Dim Slice index
for ii = 1:(numRows * numRows)
    conA = conA + 1;
    mA(conA, itmIdx:numRows:(itmIdx + ((numRows - 1) * numRows))) = 1;
    if(mod(itmIdx, numRows) == 0)
        itmIdx = (itmIdx - numRows + 1) + (numRows * numRows);
    else
        itmIdx = itmIdx + 1;
    end
end

%?%?%?
% Can you do it using `kron()`?

% Depth Slice Constraits
itmIdx = 1; %<! First item in Column / Row / 3rd Dim Slice index
for ii = 1:(numRows * numRows)
    conA = conA + 1;
    % mA(conA, itmIdx:(numRows * numRows):end) = 1;
    mA(conA, itmIdx:(numRows * numRows):(itmIdx + ((numRows - 1) * numRows * numRows))) = 1;
    itmIdx = itmIdx + 1;
end

% Sub Grid Constraits
itmIdx = 0; %<! First item in Column / Row / 3rd Dim Slice index
for kk = 1:numRows
    for nn = 0:3:6
        for mm = 0:3:6
            conA = conA + 1;
            for jj = 1:3
                for ii = 1:3
                    jn = jj + nn;
                    im = ii + mm;
                    itmIdx = ((kk - 1) * (numRows ^ 2)) + ((jn - 1) * numRows) + im;
                    mA(conA, itmIdx) = 1;
                end
            end
        end
    end
end

% Equality Const
vB = ones(numConst, 1);

vL = zeros(numVar, 1); %<! Lower Bound - 0
vU = ones(numVar, 1); %<! Upper Bound - 1

% Set vL according to input data (Clues)
for ii = 1:numClues
    clueIdx = mB(ii, 1) + ((mB(ii, 2) - 1) * numRows) + ((mB(ii, 3) - 1) * numRows * numRows);
    vL(clueIdx) = 1;
end



%% Linear Programming Optimization
% Using MATLAB's `linprog()`

sSolverOpt = optimoptions('linprog', 'Display', 'none');
vX = linprog(vF, [], [], mA, vB, vL, vU, sSolverOpt);

[~, mS] = max(reshape(vX, numRows, numRows, numRows), [], 3); %<! Solution


%% Integer Linear Programming Optimization
% Using MATLAB's intlinprog()`

% vIntFlag = 1:numVar;
% 
% sSolverOpt = optimoptions('intlinprog', 'Display', 'none');
% vX = intlinprog(vF, vIntFlag, [], [], mA, vB, vL, vU, sSolverOpt);
% 
% [~, mS] = max(reshape(vX, numRows, numRows, numRows), [], 3); %<! Solution


%% Display Results

[mJ, mI] = meshgrid(1:numRows);

DrawSudokuBoard([mI(:), mJ(:), mS(:)], numRows);



%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

