import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def get_config():
    name = 'cifar100'
    base = pd.DataFrame(ParameterGrid({'activation': ['relu'],
                                       'lr': [0.001, 0.0001],
                                       'batch_size': [128],
                                       'depth': [10, 20, 30],
                                       'width': [2, 8, 16, 32, 64, 96, 192]
                                       }))

    baseline = pd.DataFrame(ParameterGrid({
        'lambda': [0],
        'aggr': ['norm']
    }))

    norm = pd.DataFrame(ParameterGrid({
        'lambda': [0.001, 0.1],
        'aggr': ['norm']
    }))

    mean = pd.DataFrame(ParameterGrid({
        'lambda': [0.1, 1],
        'aggr': ['mean']
    }))

    baseline = base.merge(baseline, how='cross')
    norm = base.merge(norm, how='cross')
    mean = base.merge(mean, how='cross')
    param_grid = pd.concat((baseline, norm, mean))
    param_grid = param_grid.apply(lambda x: pd.to_numeric(x, downcast='integer', errors='ignore'))
    # Fix na
    param_grid = param_grid.fillna(np.nan).replace([np.nan], [None])

    config = {
        'name': name,
        'param_grid': param_grid,
        'dataset': 'cifar10',
        'output_dir': 'cifar10',
        'epochs': 400,
    }

    return config
