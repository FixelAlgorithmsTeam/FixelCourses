% Exp L∞ Distribution
% Plots the Exp L∞ distribution.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     24/12/2023
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

hLaplaceDist = @(vX, paramLambda) (1 / (2 * paramLambda)) * exp(-paramLambda * max(abs(vX)));



%% Display Data

vY = zeros(numPts, 1);

hF = figure('Position', figPosLarge);
hA = axes(hF, 'Units', 'pixels');
set(hA, 'NextPlot', 'add');
for ii = 1:length(vParamLambda)
    paramLambda = vParamLambda(ii);
    for jj = 1:numPts
        vY(jj) = hLaplaceDist(vX(jj), paramLambda);
    end
    hLineObj = line(vX, vY, 'DisplayName', ['\lambda = ', num2str(paramLambda, '%0.2f')]);
    set(hLineObj, 'LineWidth', lineWidthNormal, 'Color', mColorOrder(ii, :));
end
% xline(0, 'HandleVisibility', 'off');
% yline(0, 'HandleVisibility', 'off');
set(hA, 'XGrid', 'on', 'XMinorGrid', 'off');
set(hA, 'YGrid', 'on', 'YMinorGrid', 'off');
set(get(hA, 'Title'), 'String', {['Exp L∞ Distribution PDF']}, 'FontSize', fontSizeTitle);
% set(get(hA, 'XLabel'), 'String', {['Input Value']}, 'FontSize', fontSizeAxis);
% set(get(hA, 'YLabel'), 'String', {['Output Index']}, 'FontSize', fontSizeAxis);
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

