# encoding: utf-8
"""
    Utility functions for creating training data for CNN training.

    Copyright 2016 Matthias Leimeister
"""

import os
import pandas as pd
import numpy as np


def triang(start, mid, stop, equal=False):
    """
    Calculates a triangular window of the given size. Taken from
    https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

    :param start: starting bin (with value 0, included in the returned filter)
    :param mid: center bin (of height 1, unless norm is True)
    :param stop: end bin (with value 0, not included in the returned filter)
    :param equal: normalize the area of the filter to 1 [default=False]
    :return a triangular shaped filter

    """
    # height of the filter
    height = 1.
    # normalize the height
    if equal:
        height = 2. / (stop - start)
    # init the filter
    triang_filter = np.empty(stop - start)
    # rising edge
    triang_filter[:mid - start] = np.linspace(0, height, (mid - start), endpoint=False)
    # falling edge
    triang_filter[mid - start:] = np.linspace(height, 0, (stop - mid), endpoint=False)
    # return
    return triang_filter


def mel_filterbank(num_bands, fft_size, sample_rate):
    """
    Returns a filter matrix for a Mel filter bank.

    :param num_bands: number of filter bands
    :param fft_size: number of fft bins
    :param sample_rate: sample rate
    :return filterbank matrix
    """

    freq_ector = np.asarray([x * sample_rate / fft_size for x in xrange(0, fft_size)])

    frequencies = np.asarray([2595.0 * np.log10(1.0 + f / 700.0) for f in freq_ector])

    max_f = np.max(frequencies)
    min_f = np.min(frequencies)

    mel_bin_width = (max_f - min_f) / num_bands
    filterbank = np.zeros((fft_size, num_bands), dtype=np.float)

    for i in xrange(num_bands):

        idx_filter_1 = np.where(frequencies >= (i-1)*mel_bin_width + min_f)
        idx_filter_2 = np.where(frequencies <= (i+1)*mel_bin_width + min_f)
        idx_filter = np.intersect1d(idx_filter_1, idx_filter_2)

        if idx_filter.size == 0:
            continue

        start_idx = idx_filter[0]
        end_idx = idx_filter[-1]
        mid_idx = start_idx + np.floor((end_idx - start_idx) / 2)

        win = triang(start_idx, mid_idx, end_idx)
        filterbank[start_idx:start_idx+win.size, i] = win

    # return the list
    return filterbank


def get_segment_times(audio_file, annotation_folder):
    """
    Read segment start times from annotation file.
    :param audio_file: path to audio file
    :param annotation_folder: folder where annotations from SALAMI are stored
    :return: segment start times in seconds
    """

    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    # for some tracks, only one annotation is available, take first one as default
    # if there is no annotation available, store -1 as error code

    try:
        label_file = os.path.join(annotation_folder, file_name, 'parsed', 'textfile3_uppercase.txt')
        t = pd.read_table(label_file, header=None)
    except IOError:
        try:
            label_file = os.path.join(annotation_folder, file_name, 'parsed', 'textfile1_uppercase.txt')
            t = pd.read_table(label_file, header=None)
        except IOError:
            try:
                label_file = os.path.join(annotation_folder, file_name, 'parsed', 'textfile2_uppercase.txt')
                t = pd.read_table(label_file, header=None)
            except IOError:
                return -1

    if t[1].dtype == 'O':
        t = t[~(t[1].str.lower().isin(['silence', 'end']))]

    segment_times = t.iloc[:, 0].values

    return segment_times

def get_beat_times(audio_file, beats_folder, include_beat_numbers=False):
    """
    Read beat times from annotation file.
    :param audio_file: path to audio files
    :param beats_folder: folder with preanalysed beat times (in .beats.txt format per track)
    :return: beat times in seconds
    """

    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    beats_file = os.path.join(beats_folder, file_name + '.beats.txt')

    if not os.path.isfile(beats_file):
        print(f"Extracting beat times for {audio_file}")
        os.system(f"DBNDownBeatTracker single '{audio_file}' -o '{beats_file}'")

    t = pd.read_table(beats_file, header=None)

    if include_beat_numbers:
        return t[0].values, t[1].values
    else:
        return t[0].values

