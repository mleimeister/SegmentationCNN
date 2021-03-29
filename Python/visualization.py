# encoding: utf-8
"""
    Visualization functions for features and predictions.

    Copyright 2016 Matthias Leimeister
"""

import numpy as np
from feature_extraction import load_raw_features
from evaluation import post_processing
from utils import get_beat_times
import matplotlib.pyplot as plt
import pickle
import paths
import os


def visualize_predictions():
    """
    Visualize predictions resulting from a pretrained CNN model
    on the test dataset.
    """

    preds = np.load('../Data/predsTestTracks_100epochs_lr005.npy')
    data = np.load('../Data/testDataNormalized.npz')
    test_y = data['test_y']

    # load file lists and indices
    with open('../Data/fileListsAndIndex.pickle', 'rb') as f:
        train_files, train_idx, test_files, test_idx = pickle.load(f)

    for i in range(len(test_files)):
        f = test_files[i]
        beat_times, beat_numbers = get_beat_times(f, paths.beats_path, include_beat_numbers=True)
        print(f)

        idx = np.where(test_idx == i)[0]
        labels = test_y[idx]

        preds_track = np.squeeze(np.asarray(preds[idx]))
        processed_preds_track = post_processing(preds_track, beat_numbers)
        with_downbeat_preds = post_processing(preds_track, beat_numbers, emphasize_downbeat=True)

        preds_track = 0.5 + 0.5 * preds_track
        processed_preds_track = 1.0 + 0.5 * processed_preds_track
        with_downbeat_preds = 1.5 + 0.5  * with_downbeat_preds
        labels *= 0.5

        plt.plot(labels)
        plt.plot(preds_track)
        plt.plot(processed_preds_track)
        plt.plot(with_downbeat_preds)
        plt.savefig(os.path.join(paths.viz_path, paths.with_suffix(test_files[i], 'svg')), dpi=400)
        plt.clf()


def visualize_training_data():
    """
    Visualize log Mel beat spectra of the training dataset.
    """

    train_features, train_labels, test_features, test_labels = load_raw_features('../Data/rawFeatures.pickle')

    for features, labels in zip(train_features, train_labels):

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.imshow(features)
        ax2.plot(labels)

        ax1.set_xlim([0, features.shape[1]])
        ax1.set_ylim([0, 80])

        ax2.set_xlim([0, features.shape[1]])
        ax2.set_ylim([0, 1])

        ax1.set_adjustable('box-forced')
        ax2.set_adjustable('box-forced')

        plt.show()


def visualize_test_data():
    """
    Visualize log Mel beat spectra of the test dataset.
    """
    train_features, train_labels, test_features, test_labels = load_raw_features('../Data/rawFeatures.pickle')

    for features, labels in zip(test_features, test_labels):

        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.imshow(features)
        ax2.plot(labels)

        ax1.set_xlim([0, features.shape[1]])
        ax1.set_ylim([0, 80])

        ax2.set_xlim([0, features.shape[1]])
        ax2.set_ylim([0, 1])

        ax1.set_adjustable('box-forced')
        ax2.set_adjustable('box-forced')

        plt.show()


if __name__ == "__main__":

    visualize_predictions()
    # visualize_test_data()
    # visualize_training_data()
