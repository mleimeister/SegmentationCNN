%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   featurePostProcessingSalami.m
%
%   Post processing of Mel spectrogram features for convolutional neural
%   network training using training tracks for Salami dataset.
%
%   (c) 2016 Matthias Leimeister
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;

conf = SegmentationCNN_config();
tatsPerBeat = conf.tatsPerBeat;
saveExt = conf.saveExt;
logScalingFactor = conf.logScalingFactor;

% load features
load(['../Data/featuresSalami' saveExt '.mat']);

% check for NANs
n = ~isnan(features(1,1,:));
features = features(:,:,n);
labels = labels(n);

% check for empty entries
n = ~(sum(sum(features(:,:,:),1),2) == 0);
features = features(:,:,n);
labels = labels(n);

% permute
n = randperm(size(features,3));
features = features(:,:,n);
labels = labels(n);

train_x = features;
train_y = labels;

clear features;
clear labels;

% make size a multiple of batch size (128)
roundSize = floor(size(train_x,3)/128)*128;
train_x = train_x(:,:,1:roundSize);
train_y = train_y(1:roundSize,:);

% transform to logarithmic magnitude
train_x = log10(1+logScalingFactor.*train_x);

% make normalization per channel
N = 10000;  % subsample to fit into memory
idx = randperm(size(train_x, 3), N);
trainFeaturesM = reshape(train_x(:,:,idx), size(train_x,1), size(train_x,2)*N);
meanPerChannel = mean(trainFeaturesM,2);
stdPerChannel = sqrt(var(trainFeaturesM, 0, 2));

train_x = train_x - repmat(meanPerChannel, [1 size(train_x,2) size(train_x,3)]);
train_x = train_x ./ repmat(stdPerChannel, [1 size(train_x,2) size(train_x,3)]);

train_x = permute(train_x, [3 1 2]);

save(['../Data/cnnTrainingDataNormalizedSalami' saveExt '.mat'], 'train_x', 'train_y', 'meanPerChannel', 'stdPerChannel', 'conf', '-v7.3');
save(['../Data/cnnNormalizationSalami' saveExt '.mat'], 'meanPerChannel', 'stdPerChannel');