function [ vY ] = CalcDerivSigmoidFun( vX )
% Calculates the Derivative of the Scaled and Translated Sigmoid Function
% The function is $ f(x) = 2 * (1 / (1 + exp(-x)) - 1 $
% Applies the function per element of teh input.
arguments(Input)
    vX (:, 1) {mustBeNumeric, mustBeFinite}
end

arguments(Output)
    vY (:, 1) {mustBeNumeric, mustBeFinite}
end

vY    = (1 ./ (1 + exp(-vX)));
vY(:) = 2 * vY .* (1 - vY);


end