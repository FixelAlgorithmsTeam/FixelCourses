function [ vG ] = CalcGradLogRegFun( vX, mA, vY )
% Calculates the Logistic Regression Objective Function Gradient
arguments(Input)
    vX (:, 1) {mustBeNumeric, mustBeFinite}
    mA (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
    vY (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal}
end

arguments(Output)
    vG (:, 1) {mustBeNumeric, mustBeFinite}
end

mAx = mA * vX;

% Hadmard Prodcut instead of the diagonal
vG = mA.' * (CalcDerivSigmoidFun(mAx) .* (CalcSigmoidFun(mAx) - vY));


end