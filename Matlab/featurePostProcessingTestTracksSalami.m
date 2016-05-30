%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   featurePostProcessingTestTracksSalami.m
%
%   Post processing of Mel spectrogram features for convolutional neural
%   network training using test tracks for Salami dataset.
%
%   (c) 2016 Matthias Leimeister
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

conf = segmentationCNN_config();
tatsPerBeat = conf.tatsPerBeat;

saveExt = conf.saveExt;

logScalingFactor = conf.logScalingFactor;

% load features
load(['../Data/featuresTestTracksSalami' saveExt '.mat']);

test_x = features;
test_y = labels;

clear features;
clear labels;

% transform to logarithmic magnitude
test_x = log10(1+logScalingFactor.*test_x);

% make normalization per channel
load(['../Data/cnnNormalizationSalami' saveExt '.mat']);
test_x = test_x - repmat(meanPerChannel, [1 size(test_x,2) size(test_x,3)]);
test_x = test_x ./ repmat(stdPerChannel, [1 size(test_x,2) size(test_x,3)]);

test_x = permute(test_x, [3 1 2]);

save(['../Data/cnnTestDataNormalizedSalami' saveExt '.mat'], 'test_x', 'test_y', 'conf', '-v7.3');