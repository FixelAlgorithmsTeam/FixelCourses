% Optimization Methods - Steepset Descent Direction
% Shows the directions of the steepest descent under different norms.
% Remarks:
%   1.  sa
% TODO:
% 	1.  ds
% Release Notes
% - 1.0.000     07/12/2023  Royi Avital
%   *   First release.
%

%% General Parameters

subStreamNumberDefault = 11;

run('InitScript.m');

figureIdx           = 0; %<! Continue from Question 1
figureCounterSpec   = '%04d';

generateFigures = ON;


%% Drawing

numPts          = 1000;
circleRadius    = 1;
dimOrder        = 2;

vCircleAngle    = linspace(0, 2 * pi, numPts);

% L1 Unit Ball
vXL1 = [1, 0, -1, 0, 1];
vYL1 = [0, -1, 0, 1, 0];

% L2 Unit Ball
vXL2 = circleRadius * cos(vCircleAngle);
vYL2 = circleRadius * sin(vCircleAngle);

% L Inf Unit Ball
vXLInf = [1, 1, -1, -1, 1];
vYLInf = [1, -1, -1, 1, 1];

vGradVec = [-0.4; 0.2];
vGradVec = (6 * rand([dimOrder, 1])) - 3;
% vGradVec = vGradVec / norm(vGradVec);

vDescentVectorL1    = -sign(vGradVec) .* (abs(vGradVec) >= max(abs(vGradVec)));
vDescentVectorL2    = -vGradVec / norm(vGradVec);
vDescentVectorLInf  = -sign(vGradVec);

% figureIdx       = figureIdx + 1;
% hFigure         = figure('Position', [100, 100, 1200, 450]);
% 
% hAxes           = subplot_tight(1, 3, 1, [0.06, 0.06]);
% set(hAxes, 'NextPlot', 'add');
% hLineSeries(1) = line(vXL1, vYL1);
% set(hLineSeries(1), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(1, :));
% hLineSeries(2) = line([0, vGradVec(1)], [0, vGradVec(2)]);
% set(hLineSeries(2), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(2, :));
% hLineSeries(3) = line(vGradVec(1), vGradVec(2));
% set(hLineSeries(3), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(3), 'Color', mColorOrder(2, :));
% hLineSeries(4) = line([0, vDescentVectorL1(1)], [0, vDescentVectorL1(2)]);
% set(hLineSeries(4), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(3, :));
% hLineSeries(5) = line(vDescentVectorL1(1), vDescentVectorL1(2));
% set(hLineSeries(5), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(5), 'Color', mColorOrder(3, :));
% set(hAxes, 'DataAspectRatio', [1, 1, 1]);
% set(hAxes, 'XGrid', 'on', 'XMinorGrid', 'on');
% set(hAxes, 'YGrid', 'on', 'YMinorGrid', 'on');
% set(hAxes, 'XLim', [-2, 2], 'YLim', [-2, 2]);
% set(get(hAxes, 'Title'), 'String', {['l_1 Steepest Descent']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['x_1']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['x_2']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend(hLineSeries([1, 2, 4]), {['l_1 Unit Ball'], ['\nabla f(x)'], [' \Delta x for l_1 Norm']}, 'Location', 'northwest');
% 
% hAxes           = subplot_tight(1, 3, 2, [0.06, 0.06]);
% set(hAxes, 'NextPlot', 'add');
% hLineSeries(1) = line(vXL2, vYL2);
% set(hLineSeries(1), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(1, :));
% hLineSeries(2) = line([0, vGradVec(1)], [0, vGradVec(2)]);
% set(hLineSeries(2), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(2, :));
% hLineSeries(3) = line(vGradVec(1), vGradVec(2));
% set(hLineSeries(3), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(3), 'Color', mColorOrder(2, :));
% hLineSeries(4) = line([0, vDescentVectorL2(1)], [0, vDescentVectorL2(2)]);
% set(hLineSeries(4), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(3, :));
% hLineSeries(5) = line(vDescentVectorL2(1), vDescentVectorL2(2));
% set(hLineSeries(5), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(5), 'Color', mColorOrder(3, :));
% set(hAxes, 'DataAspectRatio', [1, 1, 1]);
% set(hAxes, 'XGrid', 'on', 'XMinorGrid', 'on');
% set(hAxes, 'YGrid', 'on', 'YMinorGrid', 'on');
% set(hAxes, 'XLim', [-2, 2], 'YLim', [-2, 2]);
% set(get(hAxes, 'Title'), 'String', {['l_2 Steepest Descent']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['x_1']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['x_2']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend(hLineSeries([1, 2, 4]), {['l_2 Unit Ball'], ['\nabla f(x)'], [' \Delta x for l_2 Norm']}, 'Location', 'northwest');
% 
% hAxes           = subplot_tight(1, 3, 3, [0.06, 0.06]);
% set(hAxes, 'NextPlot', 'add');
% hLineSeries(1) = line(vXLInf, vYLInf);
% set(hLineSeries(1), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(1, :));
% hLineSeries(2) = line([0, vGradVec(1)], [0, vGradVec(2)]);
% set(hLineSeries(2), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(2, :));
% hLineSeries(3) = line(vGradVec(1), vGradVec(2));
% set(hLineSeries(3), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(3), 'Color', mColorOrder(2, :));
% hLineSeries(4) = line([0, vDescentVectorLInf(1)], [0, vDescentVectorLInf(2)]);
% set(hLineSeries(4), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(3, :));
% hLineSeries(5) = line(vDescentVectorLInf(1), vDescentVectorLInf(2));
% set(hLineSeries(5), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
% set(hLineSeries(5), 'Color', mColorOrder(3, :));
% set(hAxes, 'DataAspectRatio', [1, 1, 1]);
% set(hAxes, 'XGrid', 'on', 'XMinorGrid', 'on');
% set(hAxes, 'YGrid', 'on', 'YMinorGrid', 'on');
% set(hAxes, 'XLim', [-2, 2], 'YLim', [-2, 2]);
% set(get(hAxes, 'Title'), 'String', {['l_\infty Steepest Descent']}, ...
%     'FontSize', fontSizeTitle);
% set(get(hAxes, 'XLabel'), 'String', {['x_1']}, ...
%     'FontSize', fontSizeAxis);
% set(get(hAxes, 'YLabel'), 'String', {['x_2']}, ...
%     'FontSize', fontSizeAxis);
% hLegend = ClickableLegend(hLineSeries([1, 2, 4]), {['l_\infty Unit Ball'], ['\nabla f(x)'], [' \Delta x for l_\infty Norm']}, 'Location', 'northwest');
% 
% if(generateFigures == ON)
%     saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
% end


figureIdx       = figureIdx + 1;
hFigure         = figure('Position', [100, 100, 750, 750]);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
% Unit Balls
hLineSeries(1) = line(vXL1, vYL1);
set(hLineSeries(1), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(1, :));
hLineSeries(2) = line(vXL2, vYL2);
set(hLineSeries(2), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(2, :));
hLineSeries(3) = line(vXLInf, vYLInf);
set(hLineSeries(3), 'LineWidth', lineWidthNormal, 'LineStyle', ':', 'Color', mColorOrder(3, :));
% Gradient Vector
hLineSeries(4) = line([0, vGradVec(1)], [0, vGradVec(2)]);
set(hLineSeries(4), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(4, :));
hLineSeries(5) = line(vGradVec(1), vGradVec(2));
set(hLineSeries(5), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
set(hLineSeries(5), 'Color', mColorOrder(4, :));
% Descent Vector L1
hLineSeries(6) = line([0, vDescentVectorL1(1)], [0, vDescentVectorL1(2)]);
set(hLineSeries(6), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(1, :));
hLineSeries(7) = line(vDescentVectorL1(1), vDescentVectorL1(2));
set(hLineSeries(7), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
set(hLineSeries(7), 'Color', mColorOrder(1, :));
% Descent Vector L2
hLineSeries(8) = line([0, vDescentVectorL2(1)], [0, vDescentVectorL2(2)]);
set(hLineSeries(8), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(2, :));
hLineSeries(9) = line(vDescentVectorL2(1), vDescentVectorL2(2));
set(hLineSeries(9), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
set(hLineSeries(9), 'Color', mColorOrder(2, :));
% Descent Vector L Inf
hLineSeries(10) = line([0, vDescentVectorLInf(1)], [0, vDescentVectorLInf(2)]);
set(hLineSeries(10), 'LineWidth', lineWidthNormal, 'Color', mColorOrder(3, :));
hLineSeries(11) = line(vDescentVectorLInf(1), vDescentVectorLInf(2));
set(hLineSeries(11), 'LineStyle', 'none', 'Marker', '.', 'MarkerSize', 30);
set(hLineSeries(11), 'Color', mColorOrder(3, :));
% Axes
set(hAxes, 'DataAspectRatio', [1, 1, 1]);
set(hAxes, 'XGrid', 'on', 'XMinorGrid', 'on');
set(hAxes, 'YGrid', 'on', 'YMinorGrid', 'on');
set(hAxes, 'XLim', [-2, 2], 'YLim', [-2, 2]);
set(get(hAxes, 'Title'), 'String', {['Steepest Descent for L_1, L_2 and L_\infty']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['x_1']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['x_2']}, ...
    'FontSize', fontSizeAxis);
% Legend
hLegend = ClickableLegend(hLineSeries([1, 2, 3, 4, 6, 8, 10]), {['L_1 Unit Ball'], ['L_2 Unit Ball'], ['L_\infty Unit Ball'], ...
    ['\nabla f(x)'], [' \Delta x for L_1 Norm'], [' \Delta x for L_2 Norm'], [' \Delta x for L_\infty Norm']});
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    set(hFigure, 'Color', 'none');
    set(hAxes, 'Color', 'none');
    set(hLegend, 'Color', 'none');
    set(hLegend, 'TextColor', 'white');
    set(hLegend, 'LineWidth', 3);
    set(get(hAxes, 'Title'), 'Color', 'white');
    set(hAxes, 'GridColor', 'white', 'MinorGridColor', 'white');
    % saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
    % print(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.svg'], '-vector', '-dsvg');
    exportgraphics(hFigure, ['Figure', num2str(figureIdx, figureCounterSpec), '.emf'], 'BackgroundColor', 'none');
end


%% L1 Norm Projection

numRows = 2;

vD = vGradVec;

cvx_begin('quiet')
    cvx_precision('best');
    variable vX(numRows)
    minimize( norm(vX - vD) )
    subject to
        norm(vX ,1) <= 1;
cvx_end

disp(['The CVX Solver Status - ', cvx_status]);
disp(['The Optimal Value Is Given By - ', num2str(cvx_optval)]);
disp(['The Optimal Argument Is Given By - [ ', num2str(vX.'), ' ]']);

[vD, vX, vDescentVectorL1]


%% Quesiton 005

vCondNumber = [1, 10, 100, 1000, 10000];
vIterIdx    = [0:1000];

numCondNum      = length(vCondNumber);
numIterations   = length(vIterIdx);

hConvRateGeneral    = @(condNum, iterationNum) (1 - (1 / condNum)) ^ iterationNum;
hConvRateSpec       = @(condNum, iterationNum) ((1 - (1 / condNum)) / (1 + (1 / condNum))) ^ (2 * iterationNum);

mConvFactorGeneral  = zeros([numIterations, numCondNum]);
mConvFactorSpec     = zeros([numIterations, numCondNum]);

cLegendString = cell([1, (2 * numCondNum)]);

for jj = 1:numCondNum
    for ii = 1:numIterations
        mConvFactorGeneral(ii, jj)  = hConvRateGeneral(vCondNumber(jj), vIterIdx(ii));
        mConvFactorSpec(ii, jj)     = hConvRateSpec(vCondNumber(jj), vIterIdx(ii));
    end
    cLegendString{jj}               = ['Generaic Convergence - Condition Number ', num2str(vCondNumber(jj))];
    cLegendString{jj + numCondNum}  = ['Specific Convergence - Condition Number ', num2str(vCondNumber(jj))];
end

figureIdx       = figureIdx + 1;
hFigure         = figure('Position', figPosLarge);
hAxes           = axes();
set(hAxes, 'NextPlot', 'add');
hLineSeries = line(vIterIdx, mConvFactorGeneral);
set(hLineSeries, 'LineWidth', lineWidthNormal);
hLineSeries = line(vIterIdx, mConvFactorSpec);
set(hLineSeries, 'LineWidth', lineWidthNormal, 'LineStyle', ':');
set(get(hAxes, 'Title'), 'String', {['Convergence Factor for Generic and Specific Convergence Rate']}, ...
    'FontSize', fontSizeTitle);
set(get(hAxes, 'XLabel'), 'String', {['Iteration Index']}, ...
    'FontSize', fontSizeAxis);
set(get(hAxes, 'YLabel'), 'String', {['Convergence Factor']}, ...
    'FontSize', fontSizeAxis);
% Legend
hLegend = ClickableLegend(cLegendString);
set(hAxes, 'LooseInset', [0.05, 0.05, 0.05, 0.05]);

if(generateFigures == ON)
    saveas(hFigure,['Figure', num2str(figureIdx, figureCounterSpec), '.eps'], 'epsc');
end


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

