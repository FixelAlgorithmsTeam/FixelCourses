function [ mX ] = ProxGradientDescentAccel( mX, hGradFun, hProxFun, stepSize, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = ProxGradientDescent( mX, hGradFun, hProxFun, stepSize, paramLambda )
%   Optimize a composition model f(x) + \lambda g(x) where f(x) is C1
%   smooth and g(x) has a Prox function using the Proximal Gradient Method.
% Input:
%   - mX            -   Input Vector.
%                       Array of the input vector of the function.
%                       Structure: Matrix (dataDim x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hGradFun      -   Gradient Function.
%                       Function handler which evaluates the Gradient
%                       Function at a given point - hObjFun(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - hProxFun      -   Proximal Operator Function.
%                       Function handler which evaluates the Proximal
%                       Operator at a given point - hProxFun(vX, paramLambda).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - stepSize      -   Step Size.
%                       The step size used in the defailt mode (Constant).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: Positive.
%   - paramLambda   -   The Lambda Parameter.
%                       The factor of the g(x) funciton in the model.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: Non Negative.
% Output:
%   - mX            -   Input Vector.
%                       The input vector at each iteration. Shows the
%                       optimization path of the function.
%                       Structure: Matrix (dataDim x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  B
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     28/12/2023  Royi Avital RopyiAvital@yahoo.com
%       *   Updated to support complex numbers.
%   -   1.0.000     28/12/2023  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mX (:, :) {mustBeNumeric, mustBeFinite}
    hGradFun (1, 1) {mustBeA(hGradFun, 'function_handle')}
    hProxFun (1, 1) {mustBeA(hProxFun, 'function_handle')}
    stepSize (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBePositive} = 1e-5
    paramLambda (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeNonnegative} = 1
end

arguments(Output)
    mX (:, :) {mustBeNumeric, mustBeFinite}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numIterations = size(mX, 2);

paramMu = stepSize;
vX = mX(:, 1);
vG = vX;
vV = vX;

% First iteration, no acceleration
vG(:) = hGradFun(vX);
vX(:) = vX - paramMu * vG;
vX(:) = hProxFun(vX, paramMu * paramLambda);
mX(:, 2) = vX;

for ii = 3:numIterations
    vV(:) = vX + ((ii - 1) / (ii + 2)) * (vX - mX(:, ii - 2));
    vG(:) = hGradFun(vV);
    vX(:) = vV - paramMu * vG;
    vX(:) = hProxFun(vX, paramMu * paramLambda);

    mX(:, ii) = vX;
end



end

