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
import paths

from operator import itemgetter

predictions_path = '../Data/predsTestTracks_100epochs_lr005.npy'
file_list_path = '../Data/fileListsAndIndex.pickle'


def load_data(preds_file, file_lists):
    """
    Loads necessary data for evaluation.

    :param preds_file: numpy file containing CNN predictions
    :param file_lists: file list of test tracks
    :return: predictions, test file list, test index per beat frame
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

    :param preds_track: CNN predictions per beat
    :return: post-processed predictions
    """

    # smoothing
    preds_track = np.convolve(preds_track, np.hamming(4) / np.sum(np.hamming(4)), 'same')

    # emphasize peaks
    if len(preds_track) >= 32:
        preds_track = np.multiply(preds_track,
                                  np.convolve(preds_track, np.hamming(32) / np.sum(np.hamming(32)), 'same'))


    # unit maximum
    preds_track /= np.max(preds_track)

    return preds_track

def get_sort_key(item):
    return item[1]

def run_eval(f_measure_thresh):
    f_measures = []
    precisions = []
    recalls = []

    preds, test_files, test_idx = load_data(predictions_path, file_list_path)
    preds = np.reshape(preds, len(preds))

    for i, f in enumerate(test_files):
        # load annotations
        segment_times = get_segment_times(f, paths.annotations_path)

        # get beat times
        beat_times, beat_numbers = get_beat_times(f, paths.beats_path, include_beat_numbers=True)

        # get predictions for current track
        preds_track = np.squeeze(np.asarray(preds[test_idx == i]))

        # post processing
        preds_track = post_processing(preds_track)

        # insert a zero value at the beginning of the predictions to help the peak-finding algorithm
        # identify the first beat of a track

        peds_track = np.insert(preds_track, 0, 0)
        peak_loc = peakutils.indexes(preds_track, min_dist=8, thres=0.1) - 1
        pred_times = beat_times[peak_loc]

        # compute f-measure
        f_score, p, r = mir_eval.onset.f_measure(np.sort(segment_times), np.sort(pred_times), window=f_measure_thresh)

        f_measures.append(f_score)
        precisions.append(p)
        recalls.append(r)

    mean_f = np.mean(np.asarray(f_measures))
    mean_p = np.mean(np.asarray(precisions))
    mean_r = np.mean(np.asarray(recalls))

    print("mean f-Measure for {}: {}, precision: {}, recall: {}".format(f_measure_thresh, mean_f, mean_p, mean_r))
    return list(zip(test_files, f_measures, precisions, recalls))

def get_sort_key(item):
    return item[1]

if __name__ == "__main__":
    short = run_eval(0.5)
    long = run_eval(3.0)

    for i in range(len(short)):
        short[i] += long[i][1:4]

    sorted_tracks = sorted(short, key=get_sort_key)

    print("{:<20}{:4}\t{:4}\t{:4}\t{:4}\t{:4}\t{:4}".format("filename", "f0.5", "p0.5", "r0.5", "f3", "p3", "r3"))
    for track in sorted_tracks:
        print("{:<20}{:4.2}\t{:4.2}\t{:4.2}\t{:4.2}\t{:4.2}\t{:4.2}".format(*track))

