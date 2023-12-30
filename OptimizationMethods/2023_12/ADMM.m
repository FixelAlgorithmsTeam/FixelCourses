function [ mX ] = ADMM( mX, hMinFun, hProxFun, mP, paramRho, paramLambda, vZ, vW )
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
%   -   1.0.000     30/12/2023  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mX (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
    hMinFun (1, 1) {mustBeA(hMinFun, 'function_handle')}
    hProxFun (1, 1) {mustBeA(hProxFun, 'function_handle')}
    mP (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
    paramRho (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBePositive} = 1
    paramLambda (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeNonnegative} = 1
    vZ (:, :) {mustBeNumeric, mustBeFinite, mustBeReal} = zeros(size(mP, 1), 1)
    vW (:, :) {mustBeNumeric, mustBeFinite, mustBeReal} = zeros(size(mP, 1), 1)
end

arguments(Output)
    mX (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numIterations = size(mX, 2);

vX = mX(:, 1);

for ii = 2:numIterations
    vX(:) = hMinFun(vZ, vW, paramRho);
    vZ(:) = hProxFun(mP * vX + vW, paramLambda / paramRho);
    vW(:) = vW + (mP * vX - vZ);

    mX(:, ii) = vX;
end


end

