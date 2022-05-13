from sklearn.model_selection import ParameterGrid


def get_config():
    name = 'test'
    param_grid = ParameterGrid(
        {'activation': ['relu'],
         'depth': [2, 3],
         'width': [2, 4],
         'lr': [1e-4, 1e-3],
         'batch_size': [128],
         'lambda': [0],
         'aggr': ['norm']
         }
    )

    config = {
        'name': name,
        'param_grid': param_grid,
        'dataset': 'moons',
        'output_dir': 'rectangular',
        'epochs': 0,
    }

    return config
