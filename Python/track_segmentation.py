#encoding: utf-8
"""
    Compute segment boundaries of a new track using a pretrained convolutional
    neural network. Analyses beat times using the MADMOM toolbox and uses
    beat-averaged log Mel spectrograms as input to the CNN.

    Usage:
    python track_segmentation.py audio_file [output_file]

    Copyright 2016 Matthias Leimeister
"""

import os, sys
import numpy as np
import pandas as pd
from feature_extraction import compute_beat_mls, normalize_features_per_band
from evaluation import post_processing
from train_segmentation_cnn import build_model
import peakutils

normalization_path = '../Data/normalization.npz'
model_weights = '../Data/model_weights_100epochs.h5'
out_dir = '../Temp/'
num_mel_bands = 80
context_length = 65


def compute_cnn_predictions(features):

    model = build_model(num_mel_bands, context_length)
    model.load_weights(model_weights)
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    features = np.expand_dims(features, 1)
    predictions = model.predict_proba(features, batch_size=1)

    return predictions


def extract_features(audio_file, beats_file):

    t = pd.read_table(beats_file, header=None)
    beat_times = t.ix[:, 0].values

    beat_mls = compute_beat_mls(filename=audio_file, beat_times=beat_times)
    features = compute_context_windows(beat_mls)

    norm_data = np.load(normalization_path)
    mean_vec = norm_data['mean_vec']
    std_vec = norm_data['std_vec']
    features, mean_vec, std_vec = normalize_features_per_band(features, mean_vec, std_vec)

    return features, beat_times


def compute_context_windows(features):
    """
    Computes context windows from MLS features to be used as input to a CNN.
    :param features: MLS features
    :return: context windows in the form (n_windows, n_melbands, n_context)
    """

    n_preallocate = 10000

    # initialize arrays for storing context windows
    data_x = np.zeros(shape=(n_preallocate, num_mel_bands, context_length), dtype=np.float32)

    feature_count = 0
    num_beats = features.shape[1]

    for k in xrange(context_length/2, num_beats-context_length/2):

        if feature_count > n_preallocate:
            break

        next_window = features[:, k-context_length/2: k+context_length/2+1]
        data_x[feature_count, :, :] = next_window
        feature_count += 1

    data_x = data_x[:feature_count, :, :]

    return data_x


def compute_segments_from_predictions(predictions, beat_times):

    predictions = np.squeeze(predictions)
    predictions = post_processing(predictions)
    peak_loc = peakutils.indexes(predictions, min_dist=8, thres=0.05)
    segment_times = beat_times[peak_loc]

    return segment_times


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage python track_segmentation.py audio_file [output_file]")
        sys.exit(1)

    audio_file = sys.argv[1]
    print("Input file: " + audio_file)

    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    if len(sys.argv) < 3:
        output_file = os.path.join(out_dir, file_name + '.segments')
    else:
        output_file = sys.argv[2]

    if not os.path.isfile(out_dir + file_name + '.beats.txt'):
        print("Extracting beat times (this might take a while)...")
        os.system('DBNBeatTracker \'single\' ' + audio_file + ' -o ' + out_dir + file_name + '.beats.txt')

    print("Computing features")
    mls_features, beat_times = extract_features(audio_file, out_dir + file_name + '.beats.txt')

    print("Computing CNN predictions")
    predictions = compute_cnn_predictions(mls_features)

    print("Get segment times")
    segment_times = compute_segments_from_predictions(predictions, beat_times)

    print("The result has been stored in " + output_file)
    np.savetxt(output_file, segment_times, fmt='%4.2f', delimiter='\n')

