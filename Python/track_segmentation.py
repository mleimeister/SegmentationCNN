#encoding: utf-8
"""
    Compute segment boundaries of a new track using a pretrained convolutional
    neural network. Analyses beat times using the MADMOM toolbox and uses
    beat-averaged log Mel spectrograms as input to the CNN.

    Usage:
    python3 track_segmentation.py audio_file [output_file]

    Copyright 2016 Matthias Leimeister
"""

import os, sys
import numpy as np
import pandas as pd
from feature_extraction import compute_features, normalize_features_per_band
from evaluation import post_processing
from train_segmentation_cnn import build_model
import peakutils

normalization_path = '../Data/normalization.npz'
model_weights = '../Data/model_weights_100epochs_lr005.h5'
out_dir = '../Temp/'
num_mel_bands = 80
context_length = 65
padding = int(context_length / 2)


def compute_cnn_predictions(mls_features, sslm_features):
    """
    Apply pretrained CNN model to features and return predictions.
    """
    model = build_model(num_mel_bands, context_length, context_length)
    model.load_weights(model_weights)
    model.compile(loss='binary_crossentropy', optimizer='sgd')

    mls_features = np.expand_dims(mls_features, 3)
    sslm_features = np.transpose(sslm_features, (2, 0, 1))
    sslm_features = np.expand_dims(sslm_features, 3)

    predictions = model.predict([mls_features, sslm_features], batch_size=1)

    return predictions


def extract_features(audio_file, beats_file):
    """
    Extracted log-scaled Mel spectrums max-pooled across beat times.
    :param audio_file: filename of audio file
    :param beats_file: file containing beat annotations
    :return: extracted features, beat times
    """

    t = pd.read_table(beats_file, header=None)
    beat_times = t[0].values
    beat_numbers = t[1].values

    beat_mls, beat_sslm, beat_times = compute_features(audio_file)
    features = compute_context_windows(beat_mls)

    norm_data = np.load(normalization_path)
    mean_vec = norm_data['mean_vec']
    std_vec = norm_data['std_vec']
    features, mean_vec, std_vec = normalize_features_per_band(features, mean_vec, std_vec)

    return features, beat_sslm, beat_times, beat_numbers


def compute_context_windows(features):
    """
    Computes context windows from MLS features to be used as input to a CNN.

    :param features: MLS features
    :return: context windows in the form (n_windows, n_melbands, n_context)
    """

    n_preallocate = 10000

    features = np.hstack((0.001 * np.random.rand(num_mel_bands, padding), features,
                         0.001 * np.random.rand(num_mel_bands, padding)))

    # initialize arrays for storing context windows
    data_x = np.zeros(shape=(n_preallocate, num_mel_bands, context_length), dtype=np.float32)

    feature_count = 0
    num_padded_features = features.shape[1]

    for k in range(padding, num_padded_features - padding):
        if feature_count > n_preallocate:
            break

        next_window = features[:, k-padding: k+padding+1]
        data_x[feature_count, :, :] = next_window
        feature_count += 1

    data_x = data_x[:feature_count, :, :]

    return data_x


def print_predictions(p, beat_times):
    for i in range(len(p)):
        print("%i:\t%.3f\t%.1f" % (i, p[i], beat_times[i]))


def compute_segments_from_predictions(predictions, beat_times, beat_numbers):
    """
    Computes the segment times from a prediction curve and the beat times
    using peak picking.
    """
    predictions = np.squeeze(predictions)

    print("raw predicitions:")
    print_predictions(predictions, beat_times)

    predictions = post_processing(predictions, beat_numbers, emphasize_downbeat=True)

    print("after post-processing:")
    print_predictions(predictions, beat_times)

    predictions = np.insert(predictions, 0, 0)
    peak_loc = peakutils.indexes(predictions, min_dist=8, thres=0.1) - 1
    segment_times = beat_times[peak_loc]

    print("beat_num\ttime:")
    for i in peak_loc:
        print("%i\t%.2f" % (i, beat_times[i]))

    return segment_times


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage python3 track_segmentation.py audio_file [output_file]")
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
        os.system('DBNDownBeatTracker \'single\' "' + audio_file + '" -o "' + out_dir + file_name + '.beats.txt"')

    print("Computing features")
    mls_features, sslm_features, beat_times, beat_numbers = extract_features(audio_file, out_dir + file_name + '.beats.txt')

    print("Computing CNN predictions")
    predictions = compute_cnn_predictions(mls_features, sslm_features)

    print("Get segment times")
    segment_times = compute_segments_from_predictions(predictions, beat_times, beat_numbers)

    print("\n")
    for f in segment_times:
        print(f)

    print("The result has been stored in " + output_file)
    np.savetxt(output_file, segment_times, fmt='%4.2f', delimiter='\n')

