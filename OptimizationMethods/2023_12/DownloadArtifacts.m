% Download Artifacts
% A script to download artifacts for a URL.
% References:
%   1.  
% Remarks:
%   1.  B
% TODO:
% 	1.  C
% Release Notes Royi Avital RoyiAvital@yahoo.com
% - 1.0.000     20/12/2023
%   *   First release.


%% General Parameters

subStreamNumberDefault = 79;

run('InitScript.m');

figureIdx           = 0;
figureCounterSpec   = '%04d';

generateFigures = OFF;

%% Constants



%% Parameters

fileUrl = 'https://drive.google.com/uc?export=download&confirm=9iBg&id=1SIN8Er2k2gYJe2k5Mer2DrLwZZK_wykc';
fileName = 'Artifacts.zip';


%% Generate / Load Data

if(isfile(fileName))
    delete(fileName);
    pause(1);
end
filePath = websave(fileName, fileUrl);


%% Unzip Data

% Deflate the contents
unzip(fileName);
pause(2.5); %<! Release the file

% Delete file
delete(fileName);


%% Restore Defaults

% set(0, 'DefaultFigureWindowStyle', 'normal');
% set(0, 'DefaultAxesLooseInset', defaultLoosInset);

