function [ vY ] = CalcDerivSigmoidFun( vX )
% Calculates the Gaussian Distribution PDF
arguments(Input)
    vX (:, 1) {mustBeNumeric, mustBeFinite}
end

arguments(Output)
    vY (:, 1) {mustBeNumeric, mustBeFinite}
end

vY    = (1 ./ (1 + exp(-vX)));
vY(:) = 2 * vY .* (1 - vY);


end