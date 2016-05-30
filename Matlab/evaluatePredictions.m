%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   evaluatePredictions.m
% 
%   Evaluates predictions from the CNN based on the labels per tatum 
%   point. Tolerance of the f-measure can be changed with the parameter
%   'toleranceInBeats'. Visualization can be turned on/off using
%   'visualize'.
%
%   (c) 2016 Matthias Leimeister
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear all;

load('../Data_matlab/cnnTestDataNormalizedSalamibeats_16bars_f57.mat', 'test_y');
load('../Data_matlab/filesTestTracksSalami.mat');
load('../Data_matlab/predsTestTracksSalamibeats_16bars_CNN_f57.mat');

visualize = false;

fMeasure = [];
precision = [];
recall = [];

toleranceInBeats = 2;   % tolerance in beats for computing f-measure
fakeFrameRate = 2;      % assuming 120 bpm = 2 beat per second
peakPickingThreshold = 0.1;

for nTrack = 1:numel(unique(trackIndex))
    
    disp(filesTestTracks{nTrack});
    
    idx = (trackIndex == nTrack);
    
    predTrack = preds(idx);
    gtTrack = test_y(idx);    
    gtLocs = find(gtTrack==1);
    
    predTrack = conv(predTrack, hamming(4)./sum(hamming(4)), 'same');
    predTrack = max(0, predTrack.*conv(predTrack, hamming(32)./sum(hamming(32)), 'same'));
    predTrack = predTrack ./ max(predTrack);
    [~, predLocs] = findpeaks(predTrack, 'minpeakheight', peakPickingThreshold, 'minpeakdistance', 8);
    
    params = be_params;
    params.fMeasure.thresh = toleranceInBeats / fakeFrameRate;
    [f,p,r,a] = be_fMeasure(gtLocs/ fakeFrameRate, predLocs/ fakeFrameRate, params);
    disp(['fMeasure ' num2str(f) ', precision ' num2str(p) ', recall ' num2str(r)]);
    
    fMeasure = [fMeasure; f];
    precision = [precision; p];
    recall = [recall; r];
    
    if (visualize)
        hold off;
        gtLoc = find(gtTrack==1);
        [~,fn,~] = fileparts(filesTestTracks{nTrack});
        figure(1); 
        plot(predTrack), hold on;
        title([fn '.wav, f-measure ' num2str(f)], 'FontSize', 14);
        line([gtLoc gtLoc], [0 1], 'LineStyle', '-.', 'Color', 'r');
        hold off;
        xlabel('Beats'), ylabel('Prob.');
        legend('Prediction', 'Ground Truth');
        pause;
    end
end

disp(['Tolerance: ' num2str(toleranceInBeats) ' beats.']);
disp(['f-Measure: ' num2str(mean(fMeasure))]);
disp(['Precision: ' num2str(mean(precision))]);
disp(['Recall: ' num2str(mean(recall))]);



