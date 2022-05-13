import keras
from keras.datasets import mnist, cifar10, cifar100
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split

from datasets.stl10_input import load_data as load_stl10
from datasets.svhn import SVHN

import numpy as np

from datasets.delgado14a import load_uci


def load_dataset(dataset, seed=None, noise=None):
    if dataset in {'cifar10', 'mnist', 'stl10', 'svhn'}:
        if dataset == 'cifar10':
            classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            y_train = keras.utils.to_categorical(y_train, classes)
            y_test = keras.utils.to_categorical(y_test, classes)
        elif dataset == 'mnist':
            classes = 10
            img_rows, img_cols = 28, 28

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            y_train = keras.utils.to_categorical(y_train, classes)
            y_test = keras.utils.to_categorical(y_test, classes)

        elif dataset == 'cifar100':
            classes = 100

            (x_train, y_train), (x_test, y_test) = cifar100.load_data()
            y_train = keras.utils.to_categorical(y_train, classes)
            y_test = keras.utils.to_categorical(y_test, classes)
        elif dataset == 'stl10':
            classes = 10

            (x_train, y_train), (x_test, y_test) = load_stl10()
            y_train = keras.utils.to_categorical(y_train - 1, classes)
            y_test = keras.utils.to_categorical(y_test - 1, classes)
        elif dataset == 'svhn':
            classes = 10

            svhn = SVHN(use_extra=True, gray=False)
            x_train, y_train, x_test, y_test = svhn.train_data, svhn.train_labels, svhn.test_data, svhn.test_labels

            y_train = keras.utils.to_categorical(y_train, classes)
            y_test = keras.utils.to_categorical(y_test, classes)
        else:
            raise ValueError('Unknown dataset {}'.format(dataset))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif dataset == 'moons':
        X, y = make_moons(n_samples=100, shuffle=True, random_state=seed, noise=noise)
        X = X / (X.max() - X.min())

        x_train_flat, x_test_flat, y_train_flat, y_test_flat = train_test_split(X, y, test_size=0.15,
                                                                                random_state=42)
        x_train = x_train_flat[:, :, np.newaxis, np.newaxis]
        x_test = x_test_flat[:, :, np.newaxis, np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train_flat[:, np.newaxis]
        y_test = y_test_flat[:, np.newaxis]

        classes = 2

    else:  # UCI
        try:
            X, y, classes = load_uci(dataset)

        except FileNotFoundError:
            raise ValueError('Unknown dataset {}'.format(dataset))

        if classes > 2:
            y = keras.utils.to_categorical(y, classes)

        x_train_flat, x_test_flat, y_train, y_test = train_test_split(X, y, test_size=0.15,
                                                                      random_state=42)
        x_train = x_train_flat[:, :, np.newaxis, np.newaxis]
        x_test = x_test_flat[:, :, np.newaxis, np.newaxis]
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255


    return (x_train, y_train), (x_test, y_test), classes
