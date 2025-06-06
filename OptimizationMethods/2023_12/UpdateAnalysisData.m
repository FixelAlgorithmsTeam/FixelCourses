function [ mObjFunValMse, mSolSe ] = UpdateAnalysisData( mObjFunValMse, mSolSe, mX, hObjFun, sCvxSol, solverIdx )
% ----------------------------------------------------------------------------------------------- %
% Remarks:
%   1.  T
% Known Issues:
%   1.  A
% TODO:
%   1.  A
% Release Notes:
%   -   1.1.000     26/12/2020
%       *   Using MSE / Squared Norm.
%   -   1.0.000     23/11/2016
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

cvxOptVal   = sCvxSol.cvxOptVal;
vXCvx       = sCvxSol.vXCvx;

numIterations = size(mSolSe, 1);

for ii = 1:numIterations

    mObjFunValMse(ii, solverIdx)    = (hObjFun(mX(:, ii)) - cvxOptVal) ^ 2;
    mSolSe(ii, solverIdx)           = sum((mX(:, ii) - vXCvx) .^ 2);
    % mSolSe(ii, solverIdx)          = norm(mX(:, ii) - vXCvx, 'inf') ^ 2;

end


end

