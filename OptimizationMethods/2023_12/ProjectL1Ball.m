function [ vX ] = ProjectL1Ball( vY, ballRadius )
% ----------------------------------------------------------------------------------------------- %
% [ vX ] = ProjectSimplexExact( vY, ballRadius )
%   Solving the Orthogonal Projection Problem of the input vector onto the
%   L1 Ball using Dual Function and exact solution by solving linear
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
% Output:
%   - vX            -   Output Vector.
%                       The projection of the Input Vector onto the Simplex
%                       Ball.
%                       Structure: Vector (Column).
%                       Type: 'Single' / 'Double'.
%                       Range: (-inf, inf).
% References
%   1.  A
% Remarks:
%   1.  The solver finds 2 points which one is positive and the other is
%       negative. Then, since the objective function is linear, finds the
%       exact point where the linear function has value of zero.
% TODO:
%   1.  U.
% Release Notes:
%   -   1.0.001     02/10/2024  Royi Avital
%       *   Fix the case `paramLambda` becomes a vector if not a single
%           unique value is verified.
%   -   1.0.000     14/04/2020  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

numElements = size(vY, 1);

if(sum(abs(vY)) <= ballRadius)
    % The input is already within the L1 Ball.
    vX = vY;
    return;
end

vZ = sort(abs(vY), 'ascend');

vParamLambda    = [0; vZ; vZ(numElements) + ballRadius];
hObjFun         = @(paramLambda) sum( max(vZ - paramLambda, 0) ) - ballRadius;

vObjVal = zeros(numElements + 2, 1);
for ii = 1:(numElements + 2)
	vObjVal(ii) = hObjFun(vParamLambda(ii));
end

if(any(vObjVal == 0))
    paramLambda = vParamLambda(vObjVal == 0); %<! Might be more than 1 case
    paramLambda = paramLambda(1); %<! Guarantees scalar in case more than 1 element vanishes
else
    % Working on when an Affine Function have the value zero
    valX1Idx = find(vObjVal > 0, 1, 'last');
    valX2Idx = find(vObjVal < 0, 1, 'first');

    valX1 = vParamLambda(valX1Idx);
    valX2 = vParamLambda(valX2Idx);
    valY1 = vObjVal(valX1Idx);
    valY2 = vObjVal(valX2Idx);

    paramA      = (valY2 - valY1) / (valX2 - valX1);
    paramB      = valY1 - (paramA * valX1);
    paramLambda = -paramB / paramA;
end

vX = sign(vY) .* max(abs(vY) - paramLambda, 0);


end

