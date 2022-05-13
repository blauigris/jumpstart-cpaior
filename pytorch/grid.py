import importlib
import os
from pathlib import Path

import fire
import pandas as pd
from accelerate import Accelerator
from sklearn.model_selection import ParameterGrid

from experiment.experiment import validate
from experiment.grid import run_grid


def main(config_module):
    # Create accelerator just for printing
    accelerator = Accelerator()
    # Experiment config
    config_module = config_module.replace('/', '.').replace('.py', '')
    config_module = importlib.import_module(config_module)
    config = config_module.get_config()
    accelerator.print(f'Successfully loaded {config_module} using config')
    accelerator.print_config = config.copy()
    accelerator.print_config['param_grid'] = config['param_grid']
    accelerator.print(accelerator.print_config)
    dataset = config['dataset']
    epochs = config['epochs']
    seed = config.get('seed', None)
    split_batches = config.get('split_batches', True)
    if isinstance(config['param_grid'], ParameterGrid):
        param_grid = pd.DataFrame(config['param_grid'])
    else:
        param_grid = config['param_grid']
    output_dir = config['output_dir']
    base_dir = os.environ.get('RESULTS_DIR', '/data-local/datafast1/criera/')
    base_dir = Path(base_dir) / output_dir
    if not base_dir.parent.exists():
        # We are in local
        base_dir = Path('results/') / output_dir
        accelerator.print(f'Base dir {base_dir.resolve()} not found, storing results in local results directory {base_dir.resolve()}')

    model_dir = base_dir / 'models'
    log_dir = base_dir / 'logs'
    config_dir = base_dir / 'config'
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Check if its already started
    found = validate(base_dir)
    if found.shape[0] > 0:
        param_grid = find_missing(param_grid, found)
        accelerator.print(f'FOUND {found.shape[0]} EXISTING EXPERIMENTS, remaining {param_grid.shape[0]}')
    else:
        accelerator.print('NO PREVIOUS EXPERIMENTS FOUND STARTING FROM SCRATCH')

    run_grid(param_grid=param_grid, dataset=dataset, epochs=epochs, model_dir=model_dir, log_dir=log_dir,
             verbose=2, config_dir=config_dir, notify=2, project=config['name'], grid_name=config_module.__name__,
             seed=seed, split_batches=split_batches)


def load_config_table(config_module):
    config_module = config_module.replace('/', '.').replace('.py', '')
    config_module = importlib.import_module(config_module)
    config = config_module.get_config()
    if isinstance(config['param_grid'], ParameterGrid):
        param_grid = pd.DataFrame(config['param_grid'])
    else:
        param_grid = config['param_grid']
    return param_grid


def find_missing(param_grid, found):
    found = found[found['has_model']]
    try:
        found = found.reset_index()[param_grid.columns]
        missing = pd.concat([param_grid, found]).drop_duplicates(keep=False)
    except KeyError:
        # There are some combinations that now include a new param so we run all of them
        missing = param_grid

    return missing


if __name__ == '__main__':
    fire.Fire(main)
