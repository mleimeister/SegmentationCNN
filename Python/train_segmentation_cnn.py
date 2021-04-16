# encoding: utf-8
"""
    Training of convolutional neural networks for music boundary detection.

    References: [1] Karen Ullrich et. al.: Boundary detection in music structure analysis using convolutional
                    neural networks, ISMIR 2014.

    Copyright 2016 Matthias Leimeister
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import tensorflow.keras.layers
from tensorflow.keras.models import Model


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

np.random.seed(1235)  # for reproducibility

import parameters

def load_training_data(dataset):
    """
    Loads the training dataset from a .npz file,
    assuming the variables train_x, train_y and
    train_weights are stored in there.

    :param dataset: path to the dataset
    :return train_x (n_items x mel_bands x context)
    :return train_y (n_items x 1)
    :return train_weights (n_items x 1)
    """

    data = np.load(dataset)
    return data['train_x'], data['train_sslm_x'], data['train_time_x'], data['train_y'], data['train_weights']


def load_test_data(dataset):
    """
    Loads the test dataset from a .npz file,
    assuming the variables train_x, train_y and
    train_weights are stored in there.

    :param dataset: path to the dataset
    :return test_x (n_items x mel_bands x context)
    :return test_y (n_items x 1)
    :return test_weights (n_items x 1)
    """

    data = np.load(dataset)
    return data['test_x'], data['test_sslm_x'], data['test_time_x'], data['test_y'], data['test_weights']


def build_model(mls_rows, mls_cols, sslm_shape):
    inputs = []
    merged_input = []

    if 'mls' in parameters.training_features:
        mls_input = layers.Input(shape=(mls_rows, mls_cols, 1), name='mls_input')
        mls = layers.Conv2D(16, (6, 8), activation='relu', name='mls_conv')(mls_input)
        mls = layers.MaxPooling2D(pool_size=(3, 6), name='mls_maxpool')(mls)
        merged_input.append(mls)
        inputs.append(mls_input)

    if 'sslm' in parameters.training_features:
        sslm_input = layers.Input(shape=(sslm_shape, sslm_shape, 1), name='sslm_input')
        sslm = layers.Conv2D(16, (8, 8), activation='relu', name='sslm_conv')(sslm_input)
        sslm = layers.MaxPooling2D(pool_size=(6, 6), name='sslm_maxpool')(sslm)

        merged_input.append(sslm)
        inputs.append(sslm_input)

    if len(merged_input) > 1:
        merged = layers.Concatenate(axis=1, name='mls_sslm_concat')(merged_input)
    else:
        merged = merged_input[0]

    merged = layers.Conv2D(64, (6, 3), activation='relu', name='concat_conv')(merged)
    merged = layers.Dropout(0.5, name='concat_dropout')(merged)

    merged = layers.Flatten()(merged)

    merged = layers.Dense(256, activation='relu', name='final_dense')(merged)
    merged = layers.Dropout(0.5, name='final_dropout')(merged)

    final_dense_input = [merged]
    if 'beat_numbers' in parameters.training_features:
        time_input = layers.Input(shape=(4,), name='time_input')
        time = layers.Dense(1, activation='relu', name='time_dense')(time_input)
        final_dense_input.append(time)
        inputs.append(time_input)

    if len(final_dense_input) > 1:
        merged = layers.Concatenate(name='final_concat')(final_dense_input)
    else:
        merged = final_dense_input[0]

    merged = layers.Dense(1, activation='sigmoid', name='final_sigmoid')(merged)

    return Model(inputs=inputs, outputs = merged)

def make_input(mls, sslm, time):
    input = []
    if 'mls' in parameters.training_features:
        input.append(mls)

    if 'sslm' in parameters.training_features:
        input.append(sslm)

    if 'beat_numbers' in parameters.training_features:
        input.append(time)

    return input

def train_model(batch_size=128, nb_epoch=100, save_ext='_100epochs_lr005', weights_file=None):
    """
    Trains a CNN model for music boundary detection (segmentation).

    :param batch_size: batch size for gradient descent training
    :param nb_epoch: number of epochs
    :param save_ext: extension for loading dataset and saving results
    :param weights_file: path to file with pretrained weights for continueing training
    """

    print('loading training data...')
    X_train, x_sslm_train, x_time_train, y_train, w_train = load_training_data('../Data/trainDataNormalized.npz')

    print('training data size:')
    print(X_train.shape)

    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]
    model = build_model(img_rows, img_cols, x_sslm_train.shape[1])

    p = np.random.permutation(X_train.shape[0])

    X_train = X_train[p, :, :]
    x_sslm_train = x_sslm_train[p, :, :]
    x_time_train = x_time_train[p]
    y_train = y_train[p]
    w_train = w_train[p]

    X_train = X_train.astype('float32')
    X_train = np.expand_dims(X_train, 3)
    x_sslm_train = np.expand_dims(x_sslm_train, 3)


    if weights_file is not None:
        model.load_weights(weights_file)

    sgd = SGD(lr=0.05, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)

    print('train model...')

    model.fit(x=make_input(X_train, x_sslm_train, x_time_train), y=y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True,
              verbose=1, validation_split=0.1, sample_weight=w_train, callbacks=[early_stopping])

    print('load test data...')
    X_test, x_sslm_test, x_time_test, y_test, w_test = load_test_data('../Data/testDataNormalized.npz')
    X_test = X_test.astype('float32')
    X_test = np.expand_dims(X_test, 3)
    x_sslm_test = np.expand_dims(x_sslm_test, 3)

    print('predict test data...')
    preds = model.predict(make_input(X_test, x_sslm_test, x_time_test), batch_size=1, verbose=1)

    print('saving results...')
    np.save('../Data/predsTestTracks' + save_ext + '.npy', preds)

    score = model.evaluate(make_input(X_test, x_sslm_test, x_time_test), y_test, verbose=1)
    print('Test score:', score)

    # save model
    model.save_weights('../Data/model_weights' + save_ext + '.h5', overwrite=True)


if __name__ == "__main__":
    train_model(nb_epoch=200)
