function [beatMelSpec, tatumTimes] = computeBeatMelSpec(x, beatTimes, conf)
% Computes a Mel spectrogram that is max-pooled over beats.

% load constants
targetFs = conf.targetFs;
fftSize = conf.fftSize;
hopSize = conf.hopSize;
tatsPerBeat = conf.tatsPerBeat;
framesPerSlice = conf.framesPerSlice;
halfContext = floor(framesPerSlice/2);
minMelFrequency = conf.minMelFrequency;
maxMelFrequency = conf.maxMelFrequency;
minMelBin = round((minMelFrequency / targetFs) * fftSize);
maxMelBin = round((maxMelFrequency / targetFs) * fftSize);

 % compute Mel spectrogram
X = abs(spectrogram([zeros(fftSize/2,1); x], hamming(fftSize), fftSize-hopSize));
X = conf.melFB * X(minMelBin : maxMelBin, :);

% compute sixteenth note grid based on beats and interpolated tatum
% period
tatumDiff = (1/tatsPerBeat).*diff(beatTimes);
tatumTimes = repmat(beatTimes, 1, tatsPerBeat) + [tatumDiff; tatumDiff(end)] * (0:tatsPerBeat-1);
tatumTimes = reshape(tatumTimes', numel(tatumTimes), 1);
tatumFrames = round(tatumTimes * (targetFs / hopSize));

% compute subsampled spectrogram at tatum grid
beatMelSpec = zeros(size(X, 1), length(tatumFrames));

for nFrame = 1:length(tatumFrames)-1
    if tatumFrames(nFrame) >= size(X,2) - 1
        break;
    end
    beatMelSpec(:, nFrame) = max(X(:, max(1, tatumFrames(nFrame)):min(tatumFrames(nFrame+1), size(X,2))), [], 2);
end

% padding with random noise for border convolution
beatMelSpec = [0.001 * rand(size(beatMelSpec, 1), halfContext), beatMelSpec, 0.001 * rand(size(beatMelSpec, 1), halfContext)];