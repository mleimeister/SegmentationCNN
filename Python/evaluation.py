# encoding: utf-8
"""
    Evaluation of segment boundary predictions using f-measure, precision
    and recall given a tolerance window (f_measure_thresh).

    Copyright 2016 Matthias Leimeister
"""

import numpy as np
from feature_extraction import get_segment_times, get_beat_times
import pickle
import peakutils
import mir_eval

predictions_path = '../Data/predsTestTracks.npy'
file_list_path = '../Data/fileListsAndIndex.pickle'
beats_folder_path = '../Data/salami-data-public-master/beats/'
annotations_folder_path = '../Data/salami-data-public-master/annotations/'
f_measure_thresh = 1    # tolerance window in seconds


def load_data(preds_file, file_lists):
    """
    Loads necessary data for evaluation.
    :return:
    """

    # load predictions
    preds = np.load(preds_file)

    # load file lists and indices
    with open(file_lists, 'rb') as f:
            train_files, train_idx, test_files, test_idx = pickle.load(f)

    return preds, test_files, test_idx


def post_processing(preds_track):
    """
    Post processing of prediction probabilities, applies smoothing
    window and emphasizes beats by multiplying with running avarage.
    :param preds:
    :return:
    """

    # smoothing
    preds_track = np.convolve(preds_track, np.hamming(4) / np.sum(np.hamming(4)), 'same')

    # emphasize peaks
    preds_track = np.multiply(preds_track,
                              np.convolve(preds_track, np.hamming(32) / np.sum(np.hamming(32)), 'same'))

    # unit maximum
    preds_track /= np.max(preds_track)

    return preds_track


if __name__ == "__main__":

    f_measures = []
    precisions = []
    recalls = []

    preds, test_files, test_idx = load_data(predictions_path, file_list_path)
    preds = np.reshape(preds, len(preds))

    for i, f in enumerate(test_files):

        print("Evaluating {}".format(f))

        # load annotations
        segment_times = get_segment_times(f, annotations_folder_path)

        # get beat times
        beat_times = get_beat_times(f, beats_folder_path)

        # get predictions for current track
        preds_track = np.squeeze(np.asarray(preds[test_idx == i]))

        # post processing
        preds_track = post_processing(preds_track)
        peak_loc = peakutils.indexes(preds_track, min_dist=8, thres=0.1)

        pred_times = beat_times[peak_loc]

        # compute f-measure
        f_score, p, r = mir_eval.onset.f_measure(segment_times, pred_times, window=f_measure_thresh)

        f_measures.append(f_score)
        precisions.append(p)
        recalls.append(r)

        print("f-Measure: {}, precision: {}, recall: {}".format(f_score, p, r))

    mean_f = np.mean(np.asarray(f_measures))
    mean_p = np.mean(np.asarray(precisions))
    mean_r = np.mean(np.asarray(recalls))

    print(" ")
    print("Mean scores across all test tracks:")
    print("f-Measure: {}, precision: {}, recall: {}".format(mean_f, mean_p, mean_r))


