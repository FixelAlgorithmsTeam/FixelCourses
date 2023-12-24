% L1 Prox - Soft Shrinkage / Soft Threshold Operator
% Plots the soft shrinkage / soft threshold Operator.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     21/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = ON;

%% Constants


%% Parameters

minVal = -10;
maxVal = 10;
numPts = 1000;

vParamLambda = [0; 0.5; 1; 3; 5];

% Visualization



%% Generate / Load Data

vX = linspace(minVal, maxVal, numPts);
vX = vX(:);


%% Shrinkage Operator

hL1Prox = @(vY, paramLambda) max(abs(vY) - paramLambda, 0) .* sign(vY);



%% Display Data

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
set(hA, 'NextPlot', 'add');
for ii = 1:length(vParamLambda)
    paramLambda = vParamLambda(ii);
    hLineObj = line(vX, hL1Prox(vX, paramLambda), 'DisplayName', ['\lambda = ', num2str(paramLambda, '%0.2f')]);
    set(hLineObj, 'LineWidth', lineWidthNormal, 'Color', mColorOrder(ii, :));
end
% xline(0, 'HandleVisibility', 'off');
% yline(0, 'HandleVisibility', 'off');
set(hA, 'XGrid', 'on', 'XMinorGrid', 'off');
set(hA, 'YGrid', 'on', 'YMinorGrid', 'off');
set(get(hA, 'Title'), 'String', {['1D Shrinkage Operator']}, 'FontSize', fontSizeTitle);
set(get(hA, 'XLabel'), 'String', {['Input Value']}, 'FontSize', fontSizeAxis);
set(get(hA, 'YLabel'), 'String', {['Output Index']}, 'FontSize', fontSizeAxis);
hLegend = ClickableLegend();
set(hA, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    set(hF, 'Color', 'none');
    set(hA, 'Color', 'none');
    set(hLegend, 'Color', 'none');
    set(hLegend, 'TextColor', 'white');
    set(hLegend, 'LineWidth', 3);
    set(get(hA, 'Title'), 'Color', 'white');
    set(hA, 'GridColor', 'white', 'MinorGridColor', 'white');
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
    % print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.svg'], '-vector', '-dsvg');
    exportgraphics(hF, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end


%% Auxiliary Functions


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

