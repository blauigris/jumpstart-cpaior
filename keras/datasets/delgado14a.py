import keras
from keras.utils import get_file
import numpy as np
import pandas as pd
import os


def load_uci(dataset, path='delgado14a'):
    """Loads an UCI dataset from delgado14a.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(X, y)`.
    """
    basepath = os.path.join('~/.keras/datasets', path, dataset)
    # Get data about dataset
    info = pd.read_csv(os.path.join(basepath, dataset + '.txt'), sep='= ', index_col=0, header=None, squeeze=True,
                       engine='python')

    # Load actual data
    data = []
    for i in range(int(info['n_arquivos'])):
        data_i = pd.read_csv(os.path.join(basepath, info['fich{}'.format(i + 1)]), sep='\t', index_col=0)
        assert data_i.shape[0] == int(info['n_patrons{}'.format(i + 1)])
        assert data_i.shape[1] == int(info['n_entradas']) + 1

        data.append(data_i)

    X = pd.concat(data)
    y = X.pop('clase')
    X, y = X.as_matrix(), y.as_matrix()
    classes = int(info['n_clases'])
    assert np.unique(y).shape[0] == classes

    return X, y, classes

# def lol():
#     datasets = [p for p in os.listdir(os.path.expanduser('~/.keras/datasets/delgado14a')) if not p.startswith('.')]
#     for dataset in datasets:
#         print('processing {}'.format(dataset))
#         try:
#             load_uci(dataset)
#         except FileNotFoundError as ex:
#             print(ex)
#
#
# lol()
