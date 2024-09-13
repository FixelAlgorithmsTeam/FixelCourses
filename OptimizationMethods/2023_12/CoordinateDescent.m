function [ mX ] = CoordinateDescent( mX, hGradFun, stepSizeMode, stepSize, hObjFun, paramAlpha )
% ----------------------------------------------------------------------------------------------- %
% [ mX ] = GradientDescent( mX, hGradFun, hStepSize )
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
%                       The step size used in the default mode (Constant).
%                       Structure: Scalar.
%                       Type: 'Single' / 'Double'.
%                       Range: Positive.
%   - hObjFun       -   Objective Function.
%                       Function handler which evaluates the Objective
%                       Function at a given point - hObjFun(vX).
%                       Structure: Function Handler.
%                       Type: Function Handler.
%                       Range: NA.
%   - paramAlpha    -   Step Size.
%                       The step size used in the default mode (Constant).
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
%   -   1.0.000     10/12/2017  Royi Avital
%       *   First release version.
% ----------------------------------------------------------------------------------------------- %

arguments(Input)
    mX (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
    hGradFun (1, 1) {mustBeA(hGradFun, 'function_handle')}
    stepSizeMode (1, 1) {mustBeMember(stepSizeMode, [1, 2, 3])} = 1
    stepSize (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBePositive} = 1e-5
    hObjFun (1, 1) {mustBeA(hObjFun, 'function_handle')} = @(vX) 0;
    paramAlpha (1, 1) {mustBeNumeric, mustBeFinite, mustBeReal, mustBePositive, mustBeInRange(paramAlpha, 0, 1)} = 0.5
end

arguments(Output)
    mX (:, :) {mustBeNumeric, mustBeFinite, mustBeReal}
end

FALSE   = 0;
TRUE    = 1;

OFF     = 0;
ON      = 1;

STEP_SIZE_MODE_CONSTANT     = 1;
STEP_SIZE_MODE_ADAPTIVE     = 2;
STEP_SIZE_MODE_LINE_SEARCH  = 3;

numComp         = size(mX, 1);
numIterations   = size(mX, 2);

paramMu = stepSize;

vX = mX(:, 1);
vZ = vX;

for ii = 2:numIterations
    for jj = 1:numComp
        valG = hGradFun(vX, jj); %<! Directional Derivative
        switch(stepSizeMode)
            case(STEP_SIZE_MODE_CONSTANT)
                paramMu = paramMu;
                valGG = paramMu * valG;
            case(STEP_SIZE_MODE_ADAPTIVE)
                currObjVal = hObjFun(vX);
                vZ(:)  = vX;
                vZ(jj) = vX(jj) - paramMu * valG;
                while(hObjFun(vZ) > currObjVal)
                    paramMu = paramAlpha * paramMu;
                    vZ(jj) = vX(jj) - paramMu * valG;
                end
                valGG = paramMu * valG;
                paramMu = max(1e-9, paramMu);
                paramMu = paramMu / paramAlpha;
            otherwise
                % Default until all modes are implemented
                paramMu = stepSize;
        end
        vX(jj) = vX(jj) - valGG;
        % vX(jj) = valG;
    end
    mX(:, ii) = vX;
end


end

