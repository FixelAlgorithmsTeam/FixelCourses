function [ hF, hA ] = DrawSudokuBoard( mD, numRows )

hF = figure();
hA = axes();
set(hA, 'NextPlot', 'add', 'DataAspectRatio', [1, 1, 1], ...
    'XTick', [], 'XTickLabel', [], 'YTick', [], 'YTickLabel', []);

rectangle('Position', [0, 0, numRows, numRows], 'LineWidth', 3, 'Clipping', 'off'); %<! Border
set(hA, 'XLim', [0, numRows], 'YLim', [0, numRows]);

% Cell Borders
for ii = 1:(numRows - 1)
    rectangle('Position', [ii, 0, 1, numRows], 'LineWidth', 0.75); %<! Vertical Lines
end

for ii = 1:(numRows - 1)
    rectangle('Position', [0, ii, numRows, 1], 'LineWidth', 0.75); %<! Horizontal Lines
end

% Block Borders
for ii = 0:3:(numRows - 1)
    rectangle('Position', [ii, 0, 3, numRows], 'LineWidth', 2); %<! Vertical Lines
end

for ii = 0:3:(numRows - 1)
    rectangle('Position', [0, ii, numRows, 3], 'LineWidth', 2); %<! Horizontal Lines
end

% Fill in the data
%
% The rows of mD are of the form (i, j, k) where i is the row counting from
% the top, j is the column, and k is the value. To place the entries in the
% boxes, j is the horizontal distance, numRows + 1 - i is the vertical
% distance. We subtract 0.5 to center the clue in the box.

for ii = 1:size(mD, 1)
    text(mD(ii, 2) - 0.5, (numRows + 0.5) - mD(ii, 1), num2str(mD(ii, 3)));
end


end


