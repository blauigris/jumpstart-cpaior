import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def get_config():
    name = 'cifar100'
    base = pd.DataFrame(ParameterGrid({'activation': ['relu'],
                                       'lr': [0.001],
                                       'batch_size': [128],
                                       'depth': [10],
                                       'width': [2]
                                       }))

    baseline = pd.DataFrame(ParameterGrid({
        'lambda': [0],
        'aggr': ['norm']
    }))


    param_grid = base.merge(baseline, how='cross')

    param_grid = param_grid.apply(lambda x: pd.to_numeric(x, downcast='integer', errors='ignore'))
    # Fix na
    param_grid = param_grid.fillna(np.nan).replace([np.nan], [None])

    config = {
        'name': name,
        'param_grid': param_grid,
        'dataset': 'cifar10-toy',
        'output_dir': 'cifar10',
        'epochs': 1,
    }

    return config
