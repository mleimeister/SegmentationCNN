function conf = segmentationCNN_config()
% Configuration for training a convolutional neural network for
% music boundary detection.

% Path to root directory of the code
conf.rootDir = '/home/matthias/projects/SegmentationCNN/';

% Extension for storing data that is loaded in python by keras
conf.saveExt = 'beats_16bars';

% Processing sample rate
conf.targetFs = 22050;

% Mel spectrogram parameters
conf.fftSize = 1024;
conf.hopSize = 512;
conf.numMelFilters = 80;
conf.minMelFrequency = 50;
conf.maxMelFrequency = 10000;

% Smearing of labels (consider frames next to an annotation as positive
% example)
conf.labelSmearing = 2;

% How many tatum points per beat (1 = beat, 2 = 8th note, etc.)
conf.tatsPerBeat = 1;

% Size of a context window for CNN processing (in beats)
conf.framesPerSlice = 64 + 1;

% Factor of negative examples compared to positive frames (as we cannot
% process the full dataset due to memory limiations)
conf.factorNegativeFrames = 5;

% Scaling of Mel spectrogram, applying log10(1 + scaling * x)
conf.logScalingFactor = 1000;

