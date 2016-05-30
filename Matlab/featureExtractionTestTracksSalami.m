%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   featureExtractionTestSalami.m
% 
%   Extracts Mel spectrogram slices at tatum level for the test tracks
%   of CNN based segmentation of the Salami dataset.
%
%   (c) 2016 Matthias Leimeister
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear all;

conf = SegmentationCNN_config();

conf.melFB = constructMelFilterbank(conf);

saveExt = conf.saveExt;
targetFs = conf.targetFs;
framesPerSlice = conf.framesPerSlice;
halfContext = floor(framesPerSlice/2);

numData = 100000;

features = single(zeros(80, framesPerSlice, numData));
labels = single(zeros(numData, 1));
trackIndex = zeros(numData, 1);

counter = 1;
trackCounter = 1;

% get files to process
fileID = fopen('test_tracks.txt');
filesTestTracks = textscan(fileID, '%s');
fclose(fileID);
filesTestTracks = filesTestTracks{1};

disp('Salami...');

for nFile = 1:numel(filesTestTracks)
    
    disp([num2str(nFile) ' / ' num2str(numel(filesTestTracks))]);
    
    % read samples    
    [x, fs] = wavread(filesTestTracks{nFile});

    % resample to 22.05 kHz mono
    x = mean(x, 2);
    x = resample(x, targetFs, fs);
    
    % get beats
    beatTimes = getBeatTimes(filesTestTracks{nFile}, conf);
    
     % compute beat Mel spectrogram
    [beatMelSpec, tatumTimes] = computeBeatMelSpec(x, beatTimes, conf);

    % get segment boundaries
    segmentStart = getSegmentTimes(filesTestTracks{nFile}, conf);
    
    if (segmentStart == -1)
        continue;
    end

    % map to spectrogram frames
    frameLabels = zeros(size(beatMelSpec, 2), 1);

    % for each segment start, compute the closest subsampled frame in Xmax
    for n = 1:length(segmentStart)
        [~, minIdx] = min(abs(tatumTimes - segmentStart(n)));
        frameLabels(halfContext + minIdx) = 1;    
    end

    for nWin = halfContext + 1 : size(beatMelSpec,2) - halfContext
        features(:, :, counter) = single(beatMelSpec(:, nWin + (-halfContext : halfContext)));
        labels(counter) = single(frameLabels(nWin));
        trackIndex(counter) = trackCounter;
        counter = counter + 1;
    end
    
    trackCounter = trackCounter + 1;
end


%% finalize

features(:, :, counter:end) = [];
labels(counter:end) = [];
trackIndex(counter:end) = [];

save(['../Data/featuresTestTracksSalami' saveExt '.mat'], 'features', 'labels', 'conf', '-v7.3');
save('../Data/filesTestTracksSalami.mat', 'filesTestTracks', 'trackIndex');
