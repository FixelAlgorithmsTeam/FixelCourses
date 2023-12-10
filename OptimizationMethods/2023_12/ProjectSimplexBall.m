function [ vX ] = ProjectSimplexBall( vY, ballRadius, epsVal )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectSimplexExact( vY, ballRadius, epsVal )
%   Solving the Orthogonal Projection Problem of the input vector onto the
%   Simplex Ball using Dual Function and exact solution by solving linear
%   equation.
% Input:
%   - vY            -   Input Vector.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
%   - ballRadius    -   Ball Radius.
%                       Sets the Radius of the Simplex Ball. For Unit
%                       Simplex set to 1.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
%   - epsVal        -   Tolerance.
%                       Tolerance how far a point can be outside the unit
%                       ball and still be considered within it.
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: (0, inf).
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the Simplex
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  https://math.stackexchange.com/questions/2327504.
% Remarks:
%   1.  The solver finds 2 points which one is positive and the other is
%       negative. Then, since the objective function is linear, finds the
%       exact point where the linear function has value of zero.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.1.000     14/04/2020  Royi Avital
%       *   Added input and output arguments validation.
%   -   1.0.000     14/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    vY (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeFloat}
    ballRadius (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeFloat, mustBePositive}
    epsVal (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeFloat, mustBePositive} = 1e-8;
end

arguments(Output)
    vX (:, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBeFloat}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);

if((abs(sum(vY) - ballRadius) < epsVal) && all(vY >= 0))
    % The input is already within the Simplex.
    vX = vY;
    return;
end

vZ = sort(vY, 'ascend');

vParamMu = [vZ(1) - ballRadius; vZ; vZ(numElements) + ballRadius];
hObjFun = @(paramMu) sum( max(vY - paramMu, 0) ) - ballRadius;

vObjVal = zeros(numElements + 2, 1);
for ii = 1:(numElements + 2)
	vObjVal(ii) = hObjFun(vParamMu(ii));
end

if(any(vObjVal == 0))
    paramMu = vParamMu(vObjVal == 0);
else
    % Working on when an Affine Function have the value zero
    valX1Idx = find(vObjVal > 0, 1, 'last');
    valX2Idx = find(vObjVal < 0, 1, 'first');

    valX1 = vParamMu(valX1Idx);
    valX2 = vParamMu(valX2Idx);
    valY1 = vObjVal(valX1Idx);
    valY2 = vObjVal(valX2Idx);

    paramA = (valY2 - valY1) / (valX2 - valX1);
    paramB = valY1 - (paramA * valX1);
    paramMu = -paramB / paramA;
end

vX = max(vY - paramMu, 0);


end

