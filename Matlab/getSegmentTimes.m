function segmentStart = getSegmentTimes(fileName, conf)
% Reads segment start times from Salami annotation files.

labelDir = [conf.rootDir filesep 'Data' filesep 'salami-data-public-master' filesep 'annotations'];

% get labels
try
    [~, fn, ~] = fileparts(fileName);
    labelFile = [labelDir filesep fn filesep 'parsed' filesep 'textfile1_uppercase.txt'];
    fileID = fopen(labelFile);
    C = textscan(fileID, '%f %s');
    segmentStart = C{1};
    fclose(fileID);
catch err
    disp('Could not open annotation.');
    segmentStart = -1;
end

