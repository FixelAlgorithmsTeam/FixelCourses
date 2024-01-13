function [ vX, mR ] = SequnetialLeastSquares( vX, valY, vA, mR, paramLambda )
% ----------------------------------------------------------------------------------------------- %
% [ vX, mR ] = SequnetialLeastSquares( vX, valY, vA, mR, paramLambda )
%   Optimize a smooth (C1) function using Coordinate Descent method.
% Input:
%   - mX            -   Input Vector.
%                       Array of the input vector of the function.
%                       Structure: Matrix (dataDim x numIterations).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - hGradFun      -   Gradient Function.
%                       Function handler which evaluates the Gradient
%                       Function at a given point - hGradFun(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - stepSizeMode  -   Step Size Mode.
%                       Sets the policy to calculate the step size by.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: {1, 2, 3, 4}.
%   - stepSize      -   Step Size.
%                       The step size used in the defailt mode (Constant).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: Positive.
%   - hObjFun       -   Objective Function.
%                       Function handler which evaluates the Objective
%                       Function at a given point - hObjFun(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - paramLambda   -   Forgetting Factor.
%                       The forgetting factor fo the sequential least
%                       squares.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: Positive.
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
%   1.  The input to `hGradFun()` is (vX, jj) where jj is the component
%       index.
% TODO:
%   1.  C
% Release Notes:
%   -   1.0.000     13/01/2024  Royi Avital RoyiAvital@yahoo.com
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    vX (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal}
    valY (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal}
    vA (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal}
    mR (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
    paramLambda (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeInRange(paramLambda, 0, 1)} = 1
end

arguments(Output)
    vX (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal}
    mR (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

vRA   = mR * vA;
mR(:) = (mR - ((vRA * vRA.') / (paramLambda + vA.' * vRA))) / paramLambda;

vX(:) = vX + (mR * vA * (valY - vA.' * vX));


end

