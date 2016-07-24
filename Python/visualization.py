# encoding: utf-8
"""
    Visualization functions for features and predictions.

    Copyright 2016 Matthias Leimeister
"""

import numpy as np
from feature_extraction import load_raw_features
from evaluation import post_processing
import matplotlib.pyplot as plt
import pickle


def visualize_predictions():
    """
    Visualize predictions resulting from a pretrained CNN model
    on the test dataset.
    """

    preds = np.load('../Data/predsTestTracks_100epochs_lr005.npy')
    train_features, train_labels, test_features, test_labels = load_raw_features('../Data/rawFeatures.pickle')

    data = np.load('../Data/testDataNormalized.npz')
    test_y = data['test_y']

    # load file lists and indices
    with open('../Data/fileListsAndIndex.pickle', 'rb') as f:
            train_files, train_idx, test_files, test_idx = pickle.load(f)

    for i in range(len(test_labels)):

        f = test_files[i]
        print f

        idx = np.where(test_idx == i)[0]
        labels = test_y[idx]

        preds_track = np.squeeze(np.asarray(preds[idx]))
        preds_track = post_processing(preds_track)
        preds_track = 0.5 + 0.5 * preds_track
        labels *= 0.5

        plt.plot(labels)
        plt.plot(preds_track)
        plt.show()


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
