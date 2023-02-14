clear();

PROC_TEST_DATA          = 1;
PROC_TRAIN_DATA         = 2;
PROC_VALIDATIOB_DATA    = 3;

processMode = PROC_VALIDATIOB_DATA;

switch(processMode)
    case(PROC_TEST_DATA)
        folderName  = 'Test';
        fileName    = 'TestData';
    case(PROC_TRAIN_DATA)
        folderName  = 'Train';
        fileName    = 'TrainData';
    case(PROC_VALIDATIOB_DATA)
        folderName  = 'Validation';
        fileName    = 'ValidationData';
end

vOutSize = [100, 100];

sFiles = dir([folderName, filesep(), '**', filesep(), '*.jpg']);
numFiles = length(sFiles);

mX = zeros(numFiles, 3 * prod(vOutSize), 'uint8');
vY = zeros(numFiles, 1, 'uint8');

for ii = 1:numFiles
    disp(['Processing image #', num2str(ii, '%04d'), ' out of ', num2str(numFiles), ' images.']);
    disp(['Processing image: ', sFiles(ii).name, '.']);
    mI = imread([sFiles(ii).folder, filesep(), sFiles(ii).name]);
    mI = imresize(mI, vOutSize);
    mX(ii, :) = mI(:);
    if contains(sFiles(ii).name, 'Red')
        vY(ii) = 0;
    elseif contains(sFiles(ii).name, 'Green')
        vY(ii) = 1;
    elseif contains(sFiles(ii).name, 'Blue')
        vY(ii) = 2;
    end
end

save(fileName, 'mX', 'vY');
