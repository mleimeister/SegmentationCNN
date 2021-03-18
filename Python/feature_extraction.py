# encoding: utf-8
"""
    This script contains functions for extracting low-level features for music boundary detection as
    presented in [1]. For this Mel log spectrograms (MLS) are extracted from audio files and sliced into
    context windows suitable for training convolutional neural networks (CNN). As dataset, the Internet Archive
    portion of the SALAMI dataset is used. Different to [1], the MLS features are pooled across beats instead of a
    fixed time window. Beat times have previously been extracted using the MADMOM toolbox [2] with the
    DBNBeatTracker algorithm from [3]. The extraction and storing the final features to disk can take quite a while.

    References: [1] Karen Ullrich et. al.: Boundary detection in music structure analysis using convolutional
                    neural networks, ISMIR 2014.
                [2] MADMOM - Python audio and music signal processing library, https://github.com/CPJKU/madmom
                [3] Sebastian Böck et. al.: A Multi-Model Approach to Beat Tracking Considering Heterogeneous
                    Music Styles, ISMIR 2014.

    Copyright 2016 Matthias Leimeister
"""

import librosa
import random
import pickle
import paths

import multiprocessing, logging

from utils import *
import scipy

context_length = 65         # how many beats make up a context window for the CNN
num_mel_bands = 80          # number of Mel bands
neg_frames_factor = 5       # how many more negative examples than segment boundaries
pos_frames_oversample = 5   # oversample positive frames because there are too few
mid_frames_oversample = 3   # oversample frames between segments
label_smearing = 1          # how many frames are positive examples around an annotation

random.seed(1234)           # for reproducibility
np.random.seed(1234)


def compute_beat_mls(filename, beat_times, mel_bands=num_mel_bands, fft_size=1024, hop_size=512):
    """
    Compute average Mel log spectrogram per beat given previously
    extracted beat times.

    :param filename: path to audio file
    :param beat_times: list of beat times in seconds
    :param mel_bands: number of Mel bands
    :param fft_size: FFT size
    :param hop_size: hop size for FFT processing
    :return: beat Mel spectrogram (mel_bands x frames)
    """

    computed_mls_file = paths.get_mls_path(filename)

    if os.path.exists(computed_mls_file):
        return np.load(computed_mls_file)


    if "/" in filename:
        path = filename
    else:
        path = os.path.join(paths.audio_path, filename)

    y, sr = librosa.load(path, sr=22050, mono=True)

    spec = np.abs(librosa.stft(y=y, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                               window=scipy.signal.hamming))

    mel_fb = librosa.filters.mel(sr=22050, n_fft=fft_size, n_mels=mel_bands, fmin=50, fmax=10000, htk=True)
    s = np.sum(mel_fb, axis=1)
    mel_fb = np.divide(mel_fb, s[:, np.newaxis])

    mel_spec = np.dot(mel_fb, spec)
    mel_spec = np.log10(1. + 1000. * mel_spec)

    beat_frames = np.round(beat_times * (22050. / hop_size)).astype('int')

    beat_melspec = np.max(mel_spec[:, beat_frames[0]:beat_frames[1]], axis=1)

    for k in range(1, beat_frames.shape[0]-1):
        beat_melspec = np.column_stack((beat_melspec,
            np.max(mel_spec[:, beat_frames[k]:beat_frames[k+1]], axis=1)))

    beat_melspec = np.column_stack((beat_melspec, mel_spec[:, beat_frames.shape[0]]))

    np.save(computed_mls_file, beat_melspec)

    return beat_melspec


def compute_features(logger, f, i, audio_files):
    logger.info("Track {} / {} ({})".format(i, len(audio_files), f))

    beat_times = get_beat_times(os.path.join(paths.audio_path, f), paths.beats_path)

    beat_mls = compute_beat_mls(f, beat_times)
    beat_mls /= np.max(beat_mls)
    return beat_mls, beat_times

def batch_extract_mls_and_labels(audio_files, beats_folder, annotation_folder):
    """
    Extract Mel log spectrogram features from a folder of audio files given pre-analysed
    beat times and segment boundary annotations. Return lists of feautres, label vectors
    and training weights (due to label smearing around annotations).

    :param audio_files: list of audio files to process
    :param beats_folder: folder containing extracted beat times
    :param annotation_folder: folder containing segment boundary annotations
    :return: list of MLS features, labels and training weights per track
    """

    feature_list = []
    labels_list = []
    failed_tracks_idx = []

    async_res = []

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    with multiprocessing.Pool(processes=8) as pool:
        for i, f in enumerate(audio_files):
            async_res.append(pool.apply_async(compute_features, (logger, f, i, audio_files, )))

        for i, f in enumerate(audio_files):
            beat_mls, beat_times = async_res[i].get()
            label_vec = np.zeros(beat_mls.shape[1],)
            segment_times = get_segment_times(f, paths.annotations_path)

            if isinstance(segment_times, int):
                failed_tracks_idx.append(i)
                print("Extraction failed - no annotation found for " + f)
                continue

            for segment_start in segment_times:
                closest_beat = np.argmin(np.abs(beat_times - segment_start))
                if closest_beat < len(label_vec):
                    label_vec[closest_beat] = 1.

            feature_list.append(beat_mls)
            labels_list.append(label_vec)

    return feature_list, labels_list, failed_tracks_idx


def normalize_features_per_band(features, mean_vec=None, std_vec=None, subsample=10000):
    """
    Normalize features to zero mean and unit variance per Mel band.

    :param features: features in the form (n_items, n_melbands, context_length)
    :param mean_vec: mean vector over bands (if None, it is computed from features)
    :param std_vec: standard deviation vector over bands (if None, it is computed from features)
    :param subsample: take only as many items to compute the mean and standard deviation
    :return: normalized features, mean and standard deviation vector
    """

    if mean_vec is None:
        # subsample features
        idx = random.sample(range(features.shape[0]), min(features.shape[0], subsample))
        temp_features = features[idx, :, :]

        # swap axes to (items, frames, bands)
        temp_features = np.swapaxes(temp_features, 2, 1)

        # reshape to (items * frames, bands)
        temp_features = np.reshape(temp_features,
                                   (temp_features.shape[0] * temp_features.shape[1], temp_features.shape[2]))

        # compute mean and std for every band
        mean_vec = np.mean(temp_features, axis=0)
        std_vec = np.std(temp_features, axis=0)

    features = features - mean_vec[np.newaxis, :, np.newaxis]
    features = features / std_vec[np.newaxis, :, np.newaxis]

    return features, mean_vec, std_vec


def prepare_batch_data(feature_list, labels_list, is_training=True):
    """
    Reads precomputed beat Mel spectrograms and slices them into context windows
    for CNN training. For the training set, subsampling is
    applied.

    :param feature_list: list of MLS features
    :param labels_list: list of label vectors
    :param is_training: if true, subsampling is applied
    :return: batch data in the form (n_items, n_melbands, n_context)
    """

    n_preallocate = 250000

    # initialize arrays for storing context windows
    data_x = np.zeros(shape=(n_preallocate, num_mel_bands, context_length), dtype=np.float32)
    data_y = np.zeros(shape=(n_preallocate,), dtype=np.float32)
    data_weight = np.zeros(shape=(n_preallocate,), dtype=np.float32)
    track_idx = np.zeros(shape=(n_preallocate,), dtype=int)

    feature_count = 0
    current_track = 0
    padding_length = int(context_length / 2)

    for features, labels in zip(feature_list, labels_list):

        print("Processed {} examples from {} tracks".format(feature_count, current_track+1))

        num_beats = features.shape[1]

        features = np.hstack((0.001 * np.random.rand(num_mel_bands, padding_length), features,
                             0.001 * np.random.rand(num_mel_bands, padding_length)))

        labels = np.concatenate((np.zeros(padding_length), labels, np.zeros(padding_length)), axis=0)

        if is_training is True:

            # take all positive frames
            positive_frames_idx = np.where(labels == 1)[0]

            for rep in range(pos_frames_oversample):

                for k in positive_frames_idx:

                    next_window = features[:, k - padding_length: k + padding_length + 1]
                    next_label = 1
                    next_weight = 1

                    data_x[feature_count, :, :] = next_window
                    data_y[feature_count] = next_label
                    data_weight[feature_count] = next_weight
                    track_idx[feature_count] = current_track

                    feature_count += 1

                    # apply label smearing: set labels around annotation to 1 and give them a triangular weight
                    for l in range(k - label_smearing, k + label_smearing + 1):

                        if padding_length <= l < num_beats - padding_length and l != k:

                            next_window = features[:, l-padding_length: l+padding_length+1]
                            next_label = 1
                            next_weight = 1. - np.abs(l-k) / (label_smearing + 1.)

                            data_x[feature_count, :, :] = next_window
                            data_y[feature_count] = next_label
                            data_weight[feature_count] = next_weight
                            track_idx[feature_count] = current_track

                            feature_count += 1

            # take all frames in the middle between two boundaries (typical false positives)
            mid_segment_frames_idx = (positive_frames_idx[1:] + positive_frames_idx[:-1]) / 2
            mid_segment_frames_idx = mid_segment_frames_idx.astype('int')

            for rep in range(mid_frames_oversample):

                for k in mid_segment_frames_idx:
                    for l in range(k - label_smearing, k + label_smearing + 1):

                        if padding_length <= l < num_beats - padding_length:

                            next_window = features[:, l-padding_length: l+padding_length+1]
                            next_label = 0
                            next_weight = 1

                            data_x[feature_count, :, :] = next_window
                            data_y[feature_count] = next_label
                            data_weight[feature_count] = next_weight
                            track_idx[feature_count] = current_track

                            feature_count += 1

            # sample randomly from the remaining frames
            remaining_frames_idx = [i for i in range(num_beats) if (i not in positive_frames_idx)]
            num_neg_frames = neg_frames_factor * len(positive_frames_idx) * (1 + 2 * label_smearing)

            for k in range(num_neg_frames):

                next_idx = random.sample(remaining_frames_idx, 1)[0]

                if context_length/2 <= next_idx < num_beats - padding_length:

                    next_window = features[:, next_idx-padding_length: next_idx+padding_length+1]
                    next_label = 0
                    next_weight = 1

                    data_x[feature_count, :, :] = next_window
                    data_y[feature_count] = next_label
                    data_weight[feature_count] = next_weight
                    track_idx[feature_count] = current_track

                    feature_count += 1

        else:   # test data -> extract all context windows and keep track of track indices

            for k in range(padding_length, num_beats-padding_length):

                next_window = features[:, k-padding_length: k+padding_length+1]
                next_label = labels[k]
                next_weight = 1

                data_x[feature_count, :, :] = next_window
                data_y[feature_count] = next_label
                data_weight[feature_count] = next_weight
                track_idx[feature_count] = current_track

                feature_count += 1

        current_track += 1

        if feature_count > n_preallocate:
            break

    data_x = data_x[:feature_count, :, :]
    data_y = data_y[:feature_count]
    data_weight = data_weight[:feature_count]
    track_idx = track_idx[:feature_count]

    return data_x, data_y, data_weight, track_idx


def load_raw_features(file):
    """
    Loads precomputed raw features from .pickle file.

    :param file: pickle file
    :return: training and test features and labels
    """

    with open(file, 'rb') as f:
        train_features, train_labels, test_features, test_labels = pickle.load(f)

    return train_features, train_labels, test_features, test_labels


if __name__ == "__main__":

    train_frame = pd.read_csv('../Data/train_tracks.txt', header=None)
    test_frame = pd.read_csv('../Data/test_tracks.txt', header=None)

    train_files = [train_frame.at[i, 0] for i in range(train_frame.shape[0])]
    test_files = [test_frame.at[i, 0] for i in range(test_frame.shape[0])]

    print("Extracting MLS features")

    train_features, train_labels, train_failed_idx = batch_extract_mls_and_labels(train_files,
                                                                                  paths.beats_path,
                                                                                  paths.annotations_path)

    test_features, test_labels, test_failed_idx = batch_extract_mls_and_labels(test_files,
                                                                               paths.beats_path,
                                                                               paths.annotations_path)

    print("Extracted features for {} training and {} test tracks".format(len(train_features), len(test_features)))

    # remove files where the extraction has failed (to keep track of file names later)
    for i in sorted(train_failed_idx, reverse=True):
        del train_files[i]

    for i in sorted(test_failed_idx, reverse=True):
        del test_files[i]

    with open('../Data/rawFeatures.pickle', 'wb') as f:
        pickle.dump((train_features, train_labels, test_features, test_labels), f)

    # train_features, train_labels, test_features, test_labels = load_raw_features('../Data/rawFeatures.pickle')

    train_x, train_y, train_weights, train_idx = prepare_batch_data(train_features, train_labels, is_training=True)
    test_x, test_y, test_weights, test_idx = prepare_batch_data(test_features, test_labels, is_training=False)

    train_x, mean_vec, std_vec = normalize_features_per_band(train_x)
    test_x, mean_vec, std_vec = normalize_features_per_band(test_x, mean_vec, std_vec)

    print("Prepared {} training items and {} test items".format(train_x.shape[0], test_x.shape[0]))

    # store normalized features for CNN training
    np.savez('../Data/trainDataNormalized.npz', train_x=train_x, train_y=train_y, train_weights=train_weights)
    np.savez('../Data/testDataNormalized.npz', test_x=test_x, test_y=test_y, test_weights=test_weights)
    np.savez('../Data/normalization.npz', mean_vec=mean_vec, std_vec=std_vec)

    # store file lists and index mapping to training and test data
    with open('../Data/fileListsAndIndex.pickle', 'wb') as f:
        pickle.dump((train_files, train_idx, test_files, test_idx), f)
