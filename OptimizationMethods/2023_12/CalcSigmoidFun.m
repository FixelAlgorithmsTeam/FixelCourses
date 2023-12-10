function [ vY ] = CalcSigmoidFun( vX )
% Calculates the Gaussian Distribution PDF
arguments(Input)
    vX (:, 1) {mustBeNumeric, mustBeFinite}
end

arguments(Output)
    vY (:, 1) {mustBeNumeric, mustBeFinite}
end

vY = (2 ./ (1 + exp(-vX))) - 1;


end