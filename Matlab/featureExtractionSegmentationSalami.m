%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   featureExtractionSegmentationSalami.m
% 
%   Extracts Mel spectrogram slices at tatum level as training data
%   for a convolutional neural network for music boundary detection.
%
%   (c) 2016 Matthias Leimeister
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear all;

RandStream('mcg16807', 'Seed', 0);
visualize = true;
conf = segmentationCNN_config();

% get files to process
fileID = fopen('train_tracks.txt');
files = textscan(fileID, '%s');
fclose(fileID);
files = files{1};

conf.melFB = constructMelFilterbank(conf);

targetFs = conf.targetFs;
labelSmearing = conf.labelSmearing;
tatsPerBeat = conf.tatsPerBeat;
factorNegativeFrames = conf.factorNegativeFrames;
saveExt = conf.saveExt;
framesPerSlice = conf.framesPerSlice;
halfContext = floor(framesPerSlice/2);

numData = 250000;

features = single(zeros(80, framesPerSlice, numData));
labels = single(zeros(numData, 1));

counter = 1;

for nFile = 1:numel(files)
    
    disp([num2str(nFile) ' / ' num2str(numel(files)) ', so far ' num2str(counter) ' windows']);
    
    % read samples    
    [x, fs] = wavread(files{nFile});

    % resample to 22.05 kHz mono
    x = mean(x, 2);
    x = resample(x, targetFs, fs);

    % get beats
    beatTimes = getBeatTimes(files{nFile}, conf);

    % compute beat Mel spectrogram
    [beatMelSpec, tatumTimes] = computeBeatMelSpec(x, beatTimes, conf);

    % get segment boundaries
    segmentStart = getSegmentTimes(files{nFile}, conf);
    
    if (segmentStart == -1)
        continue;
    end

    % map to spectrogram frames
    frameLabels = zeros(size(beatMelSpec, 2), 1);

    % for each segment start, compute the closest subsampled frame in Xmax
    for n = 1:length(segmentStart)
        [~, minIdx] = min(abs(tatumTimes - segmentStart(n)));
        frameLabels(halfContext + minIdx) = 1;
        
        for k = 1:labelSmearing
            frameLabels(halfContext + minIdx + k) = ...
                max(frameLabels(halfContext + minIdx + k), 1 - k/(labelSmearing+1));
            
            frameLabels(halfContext + minIdx - k) = ...
                max(frameLabels(halfContext + minIdx - k), 1 - k/(labelSmearing+1));
        end
            
    end

    if (visualize)
        figure(1), ax(1) = subplot(2,1,1); imagesc(beatMelSpec), axis xy;
        ax(2) = subplot(2,1,2); plot(frameLabels), xlim([0 length(frameLabels)]), linkaxes(ax, 'x');
        pause;
    end

    % select positive examples and apply label smearing
    positiveFrameIdx = find(frameLabels == 1);
    positiveFrames = positiveFrameIdx;
    
    for n = 1 : labelSmearing
        positiveFrames = [positiveFrames; positiveFrameIdx+n; positiveFrameIdx-n];
    end
    positiveFrames(positiveFrames < halfContext + 1) = [];
    positiveFrames(positiveFrames > size(beatMelSpec,2)-halfContext) = [];

    % take these because there are often false positives in the middle of a
    % segment
    midSegmentFrameIdx = round((positiveFrames(1:end-1) + positiveFrames(2:end)) ./ 2);
    midSegmentFrames = midSegmentFrameIdx;
    for n = 1 : labelSmearing-1
        midSegmentFrames = [midSegmentFrames; midSegmentFrameIdx+n; midSegmentFrameIdx-n];
    end
    midSegmentFrames(midSegmentFrames < halfContext + 1) = [];
    midSegmentFrames(midSegmentFrames > size(beatMelSpec,2)-halfContext) = [];
    
    negativeFrames = setdiff(halfContext+1 : size(beatMelSpec,2) - halfContext, positiveFrames);
    idx = randperm(length(negativeFrames));
    idx = idx(1 : min(length(negativeFrames), factorNegativeFrames*length(positiveFrames)));
    negativeFrames = negativeFrames(idx);
    
    keepFrames = [positiveFrames; negativeFrames'; midSegmentFrames];
   
    enoughMemory = 1;
    
    for nWin = 1 : length(keepFrames)
        
        if (counter > numData)
            enoughMemory = 0;
            break;
        end
        
        features(:, :, counter) = single(beatMelSpec(:, keepFrames(nWin) + (-halfContext : halfContext)));
        labels(counter) = single(frameLabels(keepFrames(nWin)));
        counter = counter + 1;
        
    end
    
    if (enoughMemory == 0)
        break;
    end
end

features(:, :, counter:end) = [];
labels(counter:end) = [];

save(['../Data/featuresSalami' saveExt '.mat'], 'features', 'labels', 'conf', '-v7.3');
save('../Data/testTracksSalami.mat', 'testTracks');