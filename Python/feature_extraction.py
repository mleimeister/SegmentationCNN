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
import warnings
import time
import pdb

import multiprocessing, logging
from contextlib import contextmanager

from utils import *
import scipy
import skimage.measure
from scipy.spatial import distance


context_length = 65         # how many beats make up a context window for the CNN
num_mel_bands = 80          # number of Mel bands
neg_frames_factor = 5       # how many more negative examples than segment boundaries
pos_frames_oversample = 5   # oversample positive frames because there are too few
mid_frames_oversample = 3   # oversample frames between segments
label_smearing = 1          # how many frames are positive examples around an annotation
padding_length = int(context_length / 2)

max_pool = 2

random.seed(1234)           # for reproducibility
np.random.seed(1234)

def debug_signal_handler(signal, frame):
    pdb.set_trace()


def compute_sslm(waveform, beat_times, mel_bands=num_mel_bands, fft_size=1024, hop_size=512):
    """
    Compute average Mel log spectrogram per beat given previously
    extracted beat times.

    :param waveform: raw waveform data
    :param beat_times: list of beat times in seconds
    :param mel_bands: number of Mel bands
    :param fft_size: FFT size
    :param hop_size: hop size for FFT processing
    :return: beat sslm
    """
    spec = np.abs(librosa.stft(y=waveform, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                               window=scipy.signal.hamming))

    mel_fb = librosa.filters.mel(sr=22050, n_fft=fft_size, n_mels=mel_bands, fmin=50, fmax=10000, htk=True)
    s = np.sum(mel_fb, axis=1)
    mel_fb = np.divide(mel_fb, s[:, np.newaxis])

    mel_spec = np.dot(mel_fb, spec)

    S_to_dB = librosa.power_to_db(mel_spec,ref=np.max)

    # first max-pooling: by 2.
    x_prime = skimage.measure.block_reduce(S_to_dB, (1,max_pool), np.max)

    MFCCs = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    MFCCs = MFCCs[1:,:] + 1

    # stack (bag?) two frames
    m = 2
    x = [np.roll(MFCCs,n,axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)

    x_hat_length = x_hat.shape[1]
    #Cosine distance calculation: D[N/p,L/p] matrix

    sslm_shape = context_length * 3 # because we'll max pool it down at the end

    distances = np.full((x_hat_length, sslm_shape), 1.0, dtype=np.float32) #D has as dimensions N/p and L/p
    for i in range(x_hat_length):
        for l in range(sslm_shape):
            # note that negative indices here make our matrix 'time-circular'
            cosine_dist = distance.cosine(x_hat[:,i], x_hat[:,i-(l+1)]) #cosine distance between columns i and i-L
            distances[i,l] = cosine_dist

    #Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1 #equalization factor of 10%

    epsilon_buf = np.empty((sslm_shape, sslm_shape * 2), dtype=np.float32)
    epsilon = np.empty((distances.shape[0], sslm_shape), dtype=np.float32)

    for i in range(distances.shape[0]):
        for l in range(sslm_shape):
            epsilon_buf[l] = np.concatenate((distances[i-(l+1),:], distances[i,:]))

        epsilon[i] = np.quantile(epsilon_buf, kappa, axis=1)
        for l in range(sslm_shape):
            if epsilon[i, l] == 0:
                epsilon[i,l] = 0.000000001


    #Self Similarity Lag Matrix
    sslm = scipy.special.expit(1-distances/epsilon) #aplicación de la sigmoide
    sslm = np.transpose(sslm)

    beat_frames = np.round(beat_times * (22050. / hop_size)).astype('int')
    beat_sslms = np.zeros((context_length, context_length, beat_frames.shape[0]), dtype=np.float32)

    for k in range(beat_frames.shape[0]):
        sslm_frame = beat_frames[k] // max_pool
        sslm_frame_min = sslm_frame - sslm_shape // 2
        sslm_frame_max = sslm_frame + sslm_shape // 2 + 1
        beat_sslm = np.take(sslm, range(sslm_frame_min, sslm_frame_max), mode='wrap', axis=1)
        beat_sslms[:,:,k] = skimage.measure.block_reduce(beat_sslm, (3,3), np.max)

    return beat_sslms

def compute_beat_mls(features, beat_times, mel_bands=num_mel_bands, fft_size=1024, hop_size=512):
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
    spec = np.abs(librosa.stft(y=features, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
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

    return beat_melspec

def load_waveform(filename):
    if "/" in filename:
        path = filename
    else:
        path = os.path.join(paths.audio_path, filename)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, sr=22050, mono=True)
        return y

def get_audio_cache(filename, ext):
    path = paths.get_audio_cache_path(filename, ext)

    if os.path.exists(path):
        return np.load(path)
    else:
        return None

def set_audio_cache(filename, ext, data):
    path = paths.get_audio_cache_path(filename, ext)
    np.save(path, data)

def compute_features(f):
    beat_times = get_beat_times(os.path.join(paths.audio_path, f), paths.beats_path)

    waveform = load_waveform(f)

    beat_mls = get_audio_cache(f, '.mls.npy')
    if beat_mls is None:
        beat_mls = compute_beat_mls(waveform, beat_times)
        beat_mls /= np.max(beat_mls)
        set_audio_cache(f, '.mls.npy', beat_mls)

    beat_sslm = get_audio_cache(f, '.mls_sslm.npy')

    if beat_sslm is None:
        beat_sslm = compute_sslm(waveform, beat_times)
        set_audio_cache(f, '.mls_sslm.npy', beat_sslm)

    return beat_mls, beat_sslm, beat_times

def compute_features_async(logger, f, i, audio_files):
    logger.info("Track {} / {} ({})".format(i, len(audio_files), f))

    return compute_features(f)

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
    sslm_feature_list = []
    labels_list = []
    failed_tracks_idx = []

    do_async = True
    max_tracks = None

    async_res = []

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    n_tracks = 0
    with multiprocessing.Pool(processes=8) as pool:
        if do_async:
            for i, f in enumerate(audio_files):
                async_res.append(pool.apply_async(compute_features_async, (logger, f, i, audio_files, )))

        for i, f in enumerate(audio_files):
            if do_async:
                try:
                    beat_mls, beat_sslm, beat_times = async_res[i].get()
                except Exception as inst:
                    print("error processing {}".format(f))
                    print(inst)
                    failed_tracks_idx.append(i)
                    continue
            else:
                beat_mls, beat_sslm, beat_times = compute_features_async(logger, f, i, audio_files)

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
            sslm_feature_list.append(beat_sslm)
            labels_list.append(label_vec)

            if max_tracks is not None and n_tracks > max_tracks:
                break
            n_tracks += 1

    return feature_list, sslm_feature_list, labels_list, failed_tracks_idx


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


def prepare_batch_data(feature_list, sslm_feature_list, labels_list, is_training=True):
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
    data_sslm_x = np.zeros(shape=(n_preallocate, context_length, context_length), dtype=np.float32)
    data_y = np.zeros(shape=(n_preallocate,), dtype=np.float32)
    data_weight = np.zeros(shape=(n_preallocate,), dtype=np.float32)
    track_idx = np.zeros(shape=(n_preallocate,), dtype=int)

    feature_count = 0
    current_track = 0

    for features, sslm_features, labels in zip(feature_list, sslm_feature_list, labels_list):

        print("Processed {} examples from {} tracks".format(feature_count, current_track+1))

        num_beats = features.shape[1]

        features = np.hstack((0.001 * np.random.rand(num_mel_bands, padding_length), features,
                             0.001 * np.random.rand(num_mel_bands, padding_length)))

        labels = np.concatenate((np.zeros(padding_length), labels, np.zeros(padding_length)), axis=0)

        if is_training is True:

            # take all positive frames.  these are indexes into the already padded features.
            positive_frames_idx = np.where(labels == 1)[0]

            for rep in range(pos_frames_oversample):

                for k in positive_frames_idx:

                    next_window = features[:, k - padding_length: k + padding_length + 1]
                    next_label = 1
                    next_weight = 1

                    data_x[feature_count, :, :] = next_window
                    data_sslm_x[feature_count] =  sslm_features[:, :, k - padding_length]
                    data_y[feature_count] = next_label
                    data_weight[feature_count] = next_weight
                    track_idx[feature_count] = current_track

                    feature_count += 1

                    # apply label smearing: set labels around annotation to 1 and give them a triangular weight
                    for l in range(k - label_smearing, k + label_smearing + 1):

                        # don't smear into padding.
                        if padding_length <= l < num_beats + padding_length and l != k:

                            next_window = features[:, l-padding_length: l+padding_length+1]
                            next_label = 1
                            next_weight = 1. - np.abs(l-k) / (label_smearing + 1.)

                            data_x[feature_count, :, :] = next_window
                            data_sslm_x[feature_count] =  sslm_features[:, :, l - padding_length]
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

                        if padding_length <= l < num_beats + padding_length:

                            next_window = features[:, l-padding_length: l+padding_length+1]

                            data_sslm_x[feature_count] =  sslm_features[:, :, l - padding_length]
                            data_x[feature_count, :, :] = next_window
                            data_y[feature_count] = 0
                            data_weight[feature_count] = 1
                            track_idx[feature_count] = current_track

                            feature_count += 1

            # sample randomly from the remaining frames
            remaining_frames_idx = []
            for i in range(features.shape[1]):
                if (i not in positive_frames_idx) and (padding_length <= i < features.shape[1] - padding_length):
                    remaining_frames_idx.append(i)

            num_neg_frames = neg_frames_factor * len(positive_frames_idx) * (1 + 2 * label_smearing)

            for k in range(num_neg_frames):
                next_idx = random.sample(remaining_frames_idx, 1)[0]

                next_window = features[:, next_idx-padding_length: next_idx+padding_length+1]
                next_label = 0
                next_weight = 1

                data_x[feature_count, :, :] = next_window
                data_sslm_x[feature_count] =  sslm_features[:, :, next_idx - padding_length]
                data_y[feature_count] = next_label
                data_weight[feature_count] = next_weight
                track_idx[feature_count] = current_track

                feature_count += 1

        else:   # test data -> extract all context windows and keep track of track indices
            for k in range(padding_length, num_beats + padding_length):

                next_window = features[:, k-padding_length: k+padding_length+1]
                next_label = labels[k]
                next_weight = 1

                data_x[feature_count, :, :] = next_window
                data_y[feature_count] = next_label
                data_sslm_x[feature_count] = sslm_features[:, :, k - padding_length]

                data_weight[feature_count] = next_weight
                track_idx[feature_count] = current_track

                feature_count += 1

        current_track += 1

        if feature_count > n_preallocate:
            break

    data_x = data_x[:feature_count, :, :]
    data_sslm_x = data_sslm_x[:feature_count, :, :]
    data_y = data_y[:feature_count]
    data_weight = data_weight[:feature_count]
    track_idx = track_idx[:feature_count]

    return data_x, data_sslm_x, data_y, data_weight, track_idx


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
    #import signal
    #signal.signal(signal.SIGINT, debug_signal_handler)

    train_frame = pd.read_csv('../Data/train_tracks.txt', header=None)
    test_frame = pd.read_csv('../Data/test_tracks.txt', header=None)

    train_files = [train_frame.at[i, 0] for i in range(train_frame.shape[0])]
    test_files = [test_frame.at[i, 0] for i in range(test_frame.shape[0])]

    print("Extracting MLS features")

    train_features, train_sslm_features, train_labels, train_failed_idx = batch_extract_mls_and_labels(train_files,
                                                                                  paths.beats_path,
                                                                                  paths.annotations_path)

    test_features, test_sslm_features, test_labels, test_failed_idx = batch_extract_mls_and_labels(test_files,
                                                                               paths.beats_path,
                                                                               paths.annotations_path)

    print("Extracted features for {} training and {} test tracks".format(len(train_features), len(test_features)))

    # remove files where the extraction has failed (to keep track of file names later)
    for i in sorted(train_failed_idx, reverse=True):
        del train_files[i]

    for i in sorted(test_failed_idx, reverse=True):
        del test_files[i]

    with open('../Data/rawFeatures.pickle', 'wb') as f:
        pickle.dump((train_features, train_sslm_features, train_labels, test_features, test_sslm_features, test_labels), f)

    # train_features, train_labels, test_features, test_labels = load_raw_features('../Data/rawFeatures.pickle')

    train_x, train_sslm_x, train_y, train_weights, train_idx = prepare_batch_data(train_features, train_sslm_features, train_labels, is_training=True)
    test_x, test_sslm_x, test_y, test_weights, test_idx = prepare_batch_data(test_features, test_sslm_features, test_labels, is_training=False)

    train_x, mean_vec, std_vec = normalize_features_per_band(train_x)
    test_x, mean_vec, std_vec = normalize_features_per_band(test_x, mean_vec, std_vec)

    print("Prepared {} training items and {} test items".format(train_x.shape[0], test_x.shape[0]))

    # store normalized features for CNN training
    np.savez('../Data/trainDataNormalized.npz', train_x=train_x, train_sslm_x=train_sslm_x, train_y=train_y, train_weights=train_weights)
    np.savez('../Data/testDataNormalized.npz', test_x=test_x, test_sslm_x=test_sslm_x, test_y=test_y, test_weights=test_weights)
    np.savez('../Data/normalization.npz', mean_vec=mean_vec, std_vec=std_vec)

    # store file lists and index mapping to training and test data
    with open('../Data/fileListsAndIndex.pickle', 'wb') as f:
        pickle.dump((train_files, train_idx, test_files, test_idx), f)
