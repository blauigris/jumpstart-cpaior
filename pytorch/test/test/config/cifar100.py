import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

# "the results for 20 × 8 are better than
# 10 ×16 and likewise for 20 ×32 when compared with 10 ×64"

# We test every depth in {10,20,30} with
# every width in {2,8,16,32,64,96,192}. The networks are implemented in Py-
# torch [50], with learning rates ε ∈ {0.001,0.0001} over 400 epochs, batch size
# of 128, kernel dimensions and padding as before, Kaiming uniform initializa-
# tion [24], global max-avg concat pooling before the output layer, and jumpstart
# networks with 2-norm (L2) as one aggregation function P with loss coefficient
# λ ∈{0.001,0.1} as well as the mean ( ̄x) as another P with λ ∈{0.1,1}.

def get_config():
    name = 'cifar100'
    hparams = pd.DataFrame(ParameterGrid({'activation': ['relu'],

            'lr': [0.001],
            'batch_size': [128],
            }))

    pairs = pd.DataFrame([(20, 8)], columns=['depth', 'width'])
    base = hparams.merge(pairs, how='cross')
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
        'dataset': 'cifar100',
        'output_dir': 'cifar100',
        'epochs': 400,
    }

    return config

