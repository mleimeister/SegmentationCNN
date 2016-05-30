function melFB = constructMelFilterbank(conf)
% Constructs dense mel filter bank matrix for a given configuration.

targetFs = conf.targetFs;
fftSize = conf.fftSize;
numMelFilters = conf.numMelFilters;

minMelFrequency = conf.minMelFrequency;
maxMelFrequency = conf.maxMelFrequency;

minMelBin = round((minMelFrequency / targetFs) * fftSize);
maxMelBin = round((maxMelFrequency / targetFs) * fftSize);

melFB = melfilter(numMelFilters, (minMelBin:maxMelBin)/fftSize*targetFs);
melFB = full(melFB);
for n = 1:size(melFB,1)
    melFB(n,:) = melFB(n,:)/sum(melFB(n,:));
end