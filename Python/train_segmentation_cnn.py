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
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

np.random.seed(1234)  # for reproducibility


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
    train_x = data['train_x']
    train_y = data['train_y']
    train_weights = data['train_weights']

    return train_x, train_y, train_weights


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
    test_x = data['test_x']
    test_y = data['test_y']
    test_weights = data['test_weights']

    return test_x, test_y, test_weights


def build_model(img_rows, img_cols):

    model = Sequential()

    model.add(Convolution2D(32, 6, 8, border_mode='valid',
                            input_shape=(1, img_rows, img_cols), init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(5, 2)))
    model.add(Convolution2D(64, 4, 6, border_mode='valid', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def train_model(batch_size=128, nb_epoch=100, save_ext='_100epochs_lr005', weights_file=None):
    """
    Trains a CNN model for music boundary detection (segmentation).

    :param batch_size: batch size for gradient descent training
    :param nb_epoch: number of epochs
    :param save_ext: extension for loading dataset and saving results
    :param weights_file: path to file with pretrained weights for continueing training
    """

    print 'loading training data...'
    X_train, y_train, w_train = load_training_data('../Data/trainDataNormalized.npz')

    print 'training data size:'
    print X_train.shape

    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p, :, :]
    y_train = y_train[p]
    w_train = w_train[p]

    X_train = X_train.astype('float32')
    X_train = np.expand_dims(X_train, 1)

    img_rows = X_train.shape[2]
    img_cols = X_train.shape[3]

    model = build_model(img_rows, img_cols)

    if weights_file is not None:
        model.load_weights(weights_file)

    sgd = SGD(lr=0.05, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    print 'train model...'
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True,
              verbose=1, validation_split=0.1, sample_weight=w_train, callbacks=[early_stopping])

    print 'load test data...'
    X_test, y_test, w_test = load_test_data('../Data/testDataNormalized.npz')
    X_test = X_test.astype('float32')
    X_test = np.expand_dims(X_test, 1)

    print 'predict test data...'
    preds = model.predict_proba(X_test, batch_size=1, verbose=1)

    print 'saving results...'
    np.save('../Data/predsTestTracks' + save_ext + '.npy', preds)

    score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)
    print('Test score:', score)

    # save model
    model.save_weights('../Data/model_weights' + save_ext + '.h5', overwrite=True)


if __name__ == "__main__":
    train_model()
