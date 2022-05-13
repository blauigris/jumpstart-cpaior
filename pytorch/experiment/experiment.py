import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
# from tensorboard.compat.proto import event_pb2
# from tensorflow.python.data import TFRecordDataset
from torchinfo import summary
from tqdm import tqdm

from experiment.data import load_data, dataset_shapes
from jumpstart.metrics import JumpstartMetrics


def simple_encode_params(params):
    pairs = []
    for name, value in params.items():
        pairs.append(f'{name}:{value}')
    return '_'.join(pairs)


def load_experiment(dataset='moons', filename=None, batch_size=None, n_samples=12, random_state=1):
    if filename is None:
        if dataset == 'moons':
            filename = Path('./results/compression/moons-4999-2020-08-24-13:10:46.tar')  # Orig
            # filename = Path('./models/moons-4999-25-09-2020-15-12-56.tar') # nou

        else:
            filename = Path('./results/compression/blobs-4999-05-09-2020-17-29-26.tar')

    trainloader, testloader, input_shape, output_shape = load_data(n_samples=n_samples, dataset=dataset,
                                                                   batch_size=batch_size, random_state=random_state)

    # Original network
    checkpoint = torch.load(filename)
    model = checkpoint['model']
    print(
        f'Loaded {filename} with train acc {checkpoint["metric_data"]["train_acc"]} and '
        f'val acc train acc {checkpoint["metric_data"]["val_acc"]}')

    jumpstart_metrics = JumpstartMetrics(dataloader=trainloader, model=model)
    jumpstart_metrics.eval()

    return model, jumpstart_metrics, trainloader, testloader, filename


def count_parameters_from_architecture(depth, width, input_shape, output_shape, kernel_size=None, batchnorm=False):
    if kernel_size is None:
        if len(input_shape) > 1:
            input_dimensions = np.prod(input_shape[1:])
        else:
            input_dimensions = input_shape[0]
        # Is dense
        input_layer_params = width * (input_dimensions + 1)
        hidden_layer_params = (depth - 1) * width * (width + 1)
        output_layer_params = width * output_shape + output_shape
        trainable_params = input_layer_params + hidden_layer_params + output_layer_params

    else:
        # Is conv
        input_layer_params = width * (np.prod(kernel_size) * input_shape[0] + 1)
        hidden_layer_params = (depth - 1) * width * (np.prod(kernel_size) * width + 1)
        output_layer_params = width * 2 * output_shape + output_shape
        trainable_params = input_layer_params + hidden_layer_params + output_layer_params

    if batchnorm:
        trainable_params += depth * width * 2

    # So far all parameters are trainable
    total_params = trainable_params

    return {'trainable_params': trainable_params, 'total_params': total_params}


def count_parameters_from_model(model):
    model_summary = summary(model, verbose=0)
    return {'total_params': model_summary.total_params, 'trainable_params': model_summary.trainable_params}


def load_model_from_checkpoint(model_path):
    model_path = Path(model_path)
    if not model_path.exists() and model_path.is_dir():
        raise FileNotFoundError(f'Model file not found at {model_path}')

    model_path = Path(model_path)
    model_file = sorted(model_path.parent.glob(f'{model_path.name}*.tar'), key=os.path.getsize, reverse=True)
    if len(model_file) == 1 and model_file[0].is_file():
        model_file = model_file[0]
    else:  # Its in a directory
        model_file = sorted(model_path.glob('*.tar'), key=os.path.getsize, reverse=True)[0]

    if torch.cuda.is_available():
        checkpoint = torch.load(model_file)
    else:
        checkpoint = torch.load(model_file, map_location=torch.device('cpu'))

    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'original_model' in checkpoint:
        model = checkpoint['original_model']
    else:
        raise RuntimeError(f'No model found in {model_path}')

    return model


def extract_resurrection_metrics_from_model(model, trainloader):
    # Clean model of previous hooks
    for name, module in model.named_modules():
        for hook_id, hook in module._forward_hooks.items():  # TODO remove private API
            del module._forward_hooks[hook_id]

    jumpstart_metrics = JumpstartMetrics(trainloader, model=model, compute_partitions=False)
    jumpstart_metrics.eval()
    return jumpstart_metrics.metrics


def parse_params_from_config(filename):
    filename = sorted(filename.parent.glob(f'{filename.name}*.json'), key=lambda x: len(x.name), reverse=False)[0]

    with open(filename) as f:
        config = pd.Series(json.load(f)).rename({'lambda_': 'lambda', 'loss_mode': 'aggr'})
        config = config.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        config = config.replace({'True': True, 'False': False})
        if 'n_maxpool' in config and config['n_maxpool'] == 0:
            config['maxpool_mode'] = None
        return config


def parse_resurrection_params_from_filename(filename):
    # Grab the until the first param
    split = filename.split('_')
    activation = 'relu'
    lambda_ = float(split[-1])
    lr = float(split[-2])
    batchnorm = 'bn' in filename
    if 'resnet' in filename:
        resnet = int(split[-3])
    else:
        resnet = 0

    # TODO add stuff for other datasets
    dataset = 'tinyimagenet'
    if 'deep' in filename:
        depth = 10
    else:
        depth = 30

    params = {
        'batch_size': 256,
        'width': 64,
        'epochs': 150,
        'depth': depth,
        'activation': activation,
        'lambda': lambda_,
        'lr': lr,
        'batchnorm': batchnorm,
        'resnet': resnet,
        'aggr': 'norm',
        'schedule': None,
        'n_maxpool': 0,
        'maxpool_mode': None,
        'optimizer': 'adam',
        'dropout': False,
        'init': 'kaiming',
        'mode': 'UP',
        'negative_margin': -1,
        'dataset': dataset,
        'kernel_size': (3, 3),
        'model': 'rect',
        'backprop_mode': 'single',
        'dropout_rate': 0,
        'balance': 0.5,
        'annealing_dropout_epochs': 0
    }

    params = pd.Series(params)

    return params


def parse_simplenet_params_from_filename(filename):
    # Grab the until the first param
    name = filename[:filename.index(':')].split('_')[:-1]
    # Separate name and cut and rebuild name
    *encoded_params, timestamp = filename.split('_')[len(name):]
    name = '_'.join(name)
    # Once you have the param pairs k:v you only need to split them
    params = {'name': name}
    for param_pair in encoded_params:
        param = param_pair.split(':')
        if len(param) == 2:
            k, v = param
            # cast into the appropiate type
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
        else:
            k, v = (param[0], True)
        params[k] = v
    params = pd.Series(params)

    names = {'bs': 'batch_size',
             'la': 'lambda',
             'lm': 'aggr',
             'mm': 'maxpool_mode',
             'nm': 'n_maxpool',
             'sched': 'schedule',
             'opt': 'optimizer',
             'ar': 'aggr'
             }

    params.rename(names, inplace=True)

    if 'lambda' in params:
        lambda_, aggr = params['lambda'].split('-')
        params['lambda'], params['aggr'] = float(lambda_), aggr
    else:
        params['lambda'] = 0
        params['aggr'] = None
    if 'no-bn' in params:
        params['batchnorm'] = False
        params.drop('no-bn', inplace=True)
    else:
        params['batchnorm'] = True
    if 'no-drpt' in params:
        params['dropout'] = False
        params.drop('no-drpt', inplace=True)
    else:
        params['dropout'] = True

    # Defaults
    if 'init' not in params:
        params['init'] = 'kaiming'
    if 'mode' not in params:
        params['mode'] = 'UP'
    if 'negative_margin' not in params:
        params['negative_margin'] = -1
    if 'no-mxpl' in params:
        params['n_maxpool'] = 0
        params['maxpool_mode'] = None
    else:
        params['n_maxpool'] = 3
        params['maxpool_mode'] = 'simplenet'
    if 'lambda' not in params:
        params['lambda'] = 0
    if 'optimizer' not in params:
        params['optimizer'] = 'adam'
    if 'depth' not in params:
        params['depth'] = 13
    params['dataset'] = 'cifar10'
    params['kernel_size'] = 'simplenet'
    params['resnet'] = 0
    params['model'] = 'simplenet'
    params['dropout_rate'] = 0
    params['balance'] = 0.5
    params['annealing_dropout_epochs'] = 0
    params['backprop_mode'] = 'single'

    return params


def parse_grid_params_from_filename(filename):
    model_type, shape, *str_params = filename.split('_')
    depth, width = shape.split('x')
    depth, width = int(depth), int(width)
    params = {'model_type': model_type, 'depth': depth, 'width': width}

    if model_type == 'relu':
        params['model_type'] = 'rect'
        params['activation'] = 'relu'
    elif model_type == 'resnet':
        params['ub'] = True
        params['nm'] = 'resnet'
        params['mm'] = 'resnet'

    for p in str_params:
        k, v = p.split(':')
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        params[k] = v

    params = pd.Series(params)
    # Rename
    names = {'bs': 'batch_size',
             'la': 'lambda',
             'lm': 'aggr',
             'mm': 'maxpool_mode',
             'nm': 'n_maxpool',
             'sched': 'schedule',
             'opt': 'optimizer',
             'ar': 'aggr',
             'kd': 'activation',
             'ks': 'kernel_size',
             'ub': 'batchnorm',
             'an': 'activation',
             'it': 'init',
             'be': 'balance',
             'dr': 'dropout_rate',
             'ade': 'annealing_dropout_epochs',
             'bm': 'backprop_mode'
             }
    params.rename(names, inplace=True)

    # Defaults
    if 'init' not in params:
        params['init'] = 'kaiming'
    if 'mode' not in params:
        params['mode'] = 'UP'
    if 'negative_margin' not in params:
        params['negative_margin'] = -1
    if 'maxpool_mode' not in params:
        params['maxpool_mode'] = None
    if 'n_maxpool' not in params:
        params['n_maxpool'] = 0
    if 'lambda' not in params:
        params['lambda'] = 0
    if 'optimizer' not in params:
        params['optimizer'] = 'adam'
    if 'batchnorm' not in params:
        params['batchnorm'] = False
    if 'kernel_size' not in params:
        params['kernel_size'] = (3, 3)
    if 'dropout_rate' not in params:
        params['dropout_rate'] = 0
    if 'balance' not in params:
        params['balance'] = 0.5
    if 'annealing_dropout_epochs' not in params:
        params['annealing_dropout_epochs'] = 0
    if 'backprop_mode' not in params:
        params['backprop_mode'] = 'single'

    return params


def parse_iclr_params_from_filename(filename):
    split = str(filename).split('-')

    dataset = split[0]
    i_activation = split.index('relu')
    activation = 'relu'
    if split[i_activation + 1] == 'bn':
        batchnorm = True
    else:
        batchnorm = False

    name = '-'.join(split[:i_activation])

    # Architecture
    shape = [s for s in split if 'x' in s][0]
    depth, width = shape.split('x')
    depth, width = int(depth), int(width)

    # Kernel size
    i_ks = split.index('ks')
    if split[i_ks + 1] == 'none':
        kernel_size = None
    else:
        kernel_size = (int(split[i_ks + 1]), int(split[i_ks + 2]))

    # Lambda
    try:
        i_lambda = split.index('sr')
        lambda_ = float(split[i_lambda + 1] + '.' + split[i_lambda + 2])
    except ValueError:
        lambda_ = 0

    # Aggregation mode
    try:
        i_mode = split.index('a')
        if split[i_mode + 1] == '0' and split[i_mode + 2] == '1':
            mode = 'UP'
        elif split[i_mode + 1] == '0':
            mode = 'U'
        elif split[i_mode + 1] == '1':
            mode = 'P'
        elif split[i_mode + 1] == 'none':
            mode = 'L'
        else:
            mode = 'UP'
    except ValueError:
        mode = None

    # Batchsize
    i_bs = split.index('bs')
    batch_size = int(split[i_bs + 1])

    # LR and optimizer
    lr = float(split[i_bs - 2] + '.' + split[i_bs - 1])
    optimizer = split[i_bs - 3]

    # Initialization
    try:
        i_ki = split.index('ki')
        init = split[i_ki + 1]
        if init == 'zeros':
            dropout_rate = 0.9
            annealing_dropout_epochs = 1000
        else:
            dropout_rate = 0
            annealing_dropout_epochs = 0

    except ValueError:
        init = 'glorot'
        dropout_rate = 0
        annealing_dropout_epochs = 0

    # Balance
    try:
        i_b = split.index('b')
        balance = float(split[i_b + 1] + '.' + split[i_b + 2])
    except ValueError:
        balance = 0.5

    # Negative margin
    try:
        i_nm = split.index('nm')
        negative_margin = -float(split[i_nm + 1])
    except ValueError:
        negative_margin = -1

    params = {'dataset': dataset, 'activation': activation, 'width': width, 'depth': depth, 'lambda': lambda_,
              'mode': mode,
              'batch_size': batch_size, 'lr': lr, 'optimizer': optimizer, 'aggr': 'sum',
              'init': init, 'negative_margin': negative_margin,
              'maxpool_mode': None, 'n_maxpool': 0, 'name': name, 'schedule': None, 'batchnorm': batchnorm,
              'dropout': False, 'resnet': 0, 'kernel_size': kernel_size, 'model': 'rect',
              'backprop_mode': 'average', 'dropout_rate': dropout_rate, 'balance': balance,
              'annealing_dropout_epochs': annealing_dropout_epochs
              }

    return params


def load_event_file(event_file):
    serialized_examples = TFRecordDataset(str(event_file))
    results = []
    for serialized_example in serialized_examples:
        e = event_pb2.Event.FromString(serialized_example.numpy())
        for v in e.summary.value:
            if v.tag.endswith('loss') or v.tag.endswith('acc'):
                results.append((e.step, e.wall_time, v.tag, v.simple_value))
    return pd.DataFrame(results, columns=['epoch', 'wall_time', 'tag', 'value'])


def extract_metrics_from_summary(summary_path):
    event_files = sorted(summary_path.glob('*events*'), key=os.path.getsize, reverse=True)
    for event_file in event_files:
        try:
            summary = load_event_file(event_file.resolve())
            best, missing = get_best_from_summary(summary)
            if len(missing) == 0:
                return best

        except Exception as ex:
            if ex.__class__.__name__ == 'DataLossError':
                print(f'DataLossError at file {event_file}')
            else:
                print(f'{ex} at {event_file}')

    raise ValueError(f'Missing metric data for {summary_path}')


def validate_summary(summary_path):
    summary_path = Path(summary_path)
    event_files = sorted(summary_path.glob('*events*'), key=os.path.getsize, reverse=True)
    for event_file in event_files:
        try:
            serialized_examples = TFRecordDataset(str(event_file.resolve()))

            for serialized_example in serialized_examples:
                e = event_pb2.Event.FromString(serialized_example.numpy())
                for v in e.summary.value:
                    if v.tag.endswith('loss') or v.tag.endswith('acc'):
                        return True

        except Exception as ex:
            pass

    return False


def get_best_from_summary(summary):
    maxes = summary.groupby('tag').max()['value']
    mins = summary.groupby('tag').min()['value']

    best = {'time': pd.to_timedelta(summary['wall_time'].iloc[-1] - summary['wall_time'].iloc[0], unit='S'),
            'epochs': summary['epoch'].iloc[-1] + 1}

    missing = []
    for k in maxes.index:
        try:
            best[k] = maxes[k] if k.endswith('acc') else mins[k]
        except KeyError as ex:
            missing.append(ex.args[0])

    return best, missing


def load_summary_table(result_dir, param_format='grid', trainloader=None):
    result_dir = Path(result_dir)
    log_dir = result_dir / 'logs'
    config_dir = result_dir / 'config'
    model_dir = result_dir / 'models'
    data = []
    index = []
    if not log_dir.exists():
        raise ValueError(f'{result_dir} does not exist from {Path.cwd()}')

    if not config_dir.exists():
        print('WARNING: No config directory found')
    if not model_dir.exists():
        print('WARNING: No model checkpoint directory found')

    logs = list(log_dir.glob('[!.]*'))
    with tqdm(total=len(logs)) as pbar:
        for summary_path in logs:
            try:
                if param_format == 'grid':
                    params = parse_grid_params_from_filename(summary_path.name)
                elif param_format == 'resurrection':
                    params = parse_resurrection_params_from_filename(summary_path.name)
                elif param_format == 'simplenet':
                    params = parse_simplenet_params_from_filename(summary_path.name)
                elif param_format == 'iclr':
                    params = parse_iclr_params_from_filename(summary_path.name)
                else:
                    raise ValueError(f'Unknown param format {param_format}')
                if config_dir.exists():
                    try:
                        config_params = parse_params_from_config(config_dir / summary_path.name)
                        # Check that stored params and extracted from filename agree
                        for param_k, param_v in params.items():
                            if param_k in config_params and param_v != config_params[param_k]:
                                raise RuntimeError(
                                    f'Corrupt config {summary_path.name} for {param_k}: parsed {param_v} '
                                    f'config {config_params[param_k]}')
                        # patch dataset if missing
                        if 'dataset' not in params and 'dataset' in config_params:
                            params['dataset'] = config_params['dataset']
                    except IndexError as ex:
                        print(f'Config not found for {summary_path.name} with exception {ex}')

                # patch dataset if missing
                elif 'dataset' not in params:
                    params['dataset'] = result_dir.name.replace('grid', '').replace('-', '').replace('_', '')
                # Load metric data
                metric_data = extract_metrics_from_summary(summary_path)
                # compute parameter count
                dataset_shape = dataset_shapes[params['dataset']]
                if param_format == 'simplenet':
                    parameter_count = count_parameters_from_model(SimpleNet(input_shape=dataset_shape['input_shape'],
                                                                            classes=dataset_shape['output_shape'],
                                                                            width=params['width'],
                                                                            no_dropout=not params['dropout'],
                                                                            no_batchnorm=not params['batchnorm'],
                                                                            no_maxpool=params['n_maxpool'] == 0
                                                                            ))
                elif param_format == 'resurrection':
                    pass
                else:
                    parameter_count = count_parameters_from_architecture(depth=params['depth'], width=params['width'],
                                                                         kernel_size=params['kernel_size'],
                                                                         input_shape=dataset_shape['input_shape'],
                                                                         output_shape=dataset_shape['output_shape'],
                                                                         batchnorm=params['batchnorm'])

                if model_dir.exists():
                    try:
                        model = load_model_from_checkpoint(model_dir / summary_path.name)
                        parameter_checkpoint_count = count_parameters_from_model(model)
                        if param_format == 'resurrection':
                            parameter_count = parameter_checkpoint_count
                        else:
                            assert parameter_count == parameter_checkpoint_count

                        if trainloader:
                            resurrection_metrics = extract_resurrection_metrics_from_model(model, trainloader)
                            metric_data.update(resurrection_metrics)
                            reset_model(model)
                            resurrection_metrics_init = extract_resurrection_metrics_from_model(model, trainloader)
                            resurrection_metrics_init = {f'init_{k}': v for k, v in resurrection_metrics_init.items()}
                            metric_data.update(resurrection_metrics_init)

                    except IndexError as ex:
                        print(f'Model not found for {summary_path.name} with exception {ex}')
                    except RuntimeError as ex:
                        print(f'Failed loading model for {summary_path.name} with exception {ex} ')
                    except TypeError as ex:
                        print(f'Failed loading model probably due to accelerate wrong saving {ex}')

                params['filename'] = summary_path.name
                index.append(params)
                data.append({**metric_data, **parameter_count})
            except ValueError as ex:
                print(ex)
            pbar.update()
            pbar.set_description(summary_path.name)

    index = pd.DataFrame(index)
    table = pd.DataFrame(data, index=pd.MultiIndex.from_frame(index))

    if param_format == 'iclr':
        names = {'acc': 'train_acc', 'loss': 'train_loss',
                 'regularization_loss': 'train_lambda_sep_loss', 'val_regularization_loss': 'val_lambda_sep_loss'}
        table.rename(names, inplace=True, axis=1)
        table['train_lambda_sep_loss'].fillna(0, inplace=True)
        table['val_lambda_sep_loss'].fillna(0, inplace=True)
        table['train_loss'] = table['train_loss'] - table['train_lambda_sep_loss']
        table['val_loss'] = table['val_loss'] - table['val_lambda_sep_loss']

    return table


def extract_grids(table):
    metrics = table.columns
    table = table.reset_index(level=['depth', 'width'])
    params = pd.Index(table.index.names).drop(['name', 'filename'], errors='ignore').tolist()
    data = []
    for param_values, group_index in table.groupby(params).groups.items():
        row = dict(zip(params, param_values))
        # Deal with duplicates, keep the one with the best acc if possible
        group = table.loc[group_index]
        if 'val_acc' in group:
            group.sort_values(['depth', 'width', 'val_acc'], ascending=False, inplace=True)
        group.drop_duplicates(subset=['depth', 'width'], inplace=True)
        for metric in metrics:
            grid = group.pivot(index="width", columns="depth", values=metric)
            row[metric] = grid
        data.append(row)

    grids = pd.DataFrame.from_records(data)
    grids.set_index(params, inplace=True)
    return grids


def load_result_table(output_path):
    output_path = Path(output_path)
    results = {}
    for lr_dir in output_path.glob('[!.]*'):
        logdir = lr_dir / 'logs'
        for summary_path in logdir.glob('[!.]*'):
            params = parse_params(summary_path.name)
            try:
                results[summary_path.name] = {**params, **extract_metrics_from_summary(summary_path)}
            except ValueError as ex:
                print(ex)

    return pd.DataFrame(results).T


def highlight_best(export):
    for activation_name, activation in export.groupby(level=0):
        for metric_name, metric in activation.T.groupby('Accuracy'):
            best_idx = metric.T.idxmax()
            best_metric = metric.T.max()
            best_of_the_best = best_idx.loc[best_metric.idxmax()]
            export.loc[
                best_of_the_best, best_metric.idxmax()] = f'\\textbf{{{export.loc[best_of_the_best, best_metric.idxmax()]:.4f}}}'

    return export


def export_result_table(output_path):
    output_path = Path(output_path)
    table = load_result_table(output_path=output_path)

    table.sort_values(['activation', 'lambda', 'depth', ], inplace=True)
    table['lambda'].replace({0.001: 'Yes', 0.000: 'No'}, inplace=True)
    export = table.pivot(index='lr', columns=['activation', 'lambda', 'depth', ], values=['train_acc', 'val_acc'])

    export = export.T.unstack(0)
    # export = export.loc[['Vanilla', 'Batchnorm', 'Resnet']]
    export.index.rename([None, 'Resurrection', 'Depth'], inplace=True)
    export.columns.rename(['Learning rate', 'Accuracy'], inplace=True)
    export.columns.set_levels([['0.0001', '0.001'], ['Train', 'Val.']], inplace=True)
    export = export.apply(pd.to_numeric)
    export = highlight_best(export)
    export.to_latex(output_path / f'{output_path.name}.tex', escape=False)


def parse_params(filename):
    filename_params = filename.split('_')
    params = {}

    if 'bn' in filename and 'resnet' in filename:
        params['activation'] = 'Resnet & Batchnorm'
    elif 'bn' in filename:
        params['activation'] = 'Batchnorm'
    elif 'resnet' in filename:
        params['activation'] = 'Resnet'
    else:
        params['activation'] = 'Vanilla'
    params['depth'] = 30 if 'deep' in filename else 10
    params['lr'] = float(filename_params[-2])
    params['lambda'] = float(filename_params[-1])

    return params


def validate(result_dir, param_format='grid', purge=False):
    result_dir = Path(result_dir)
    log_dir = result_dir / 'logs'
    config_dir = result_dir / 'config'
    model_dir = result_dir / 'models'
    data = []
    index = []
    if not log_dir.exists():
        raise ValueError(f'{result_dir} does not exist')

    if not config_dir.exists():
        print('WARNING: No config directory found')
    if not model_dir.exists():
        print('WARNING: No model checkpoint directory found')

    logs = list(log_dir.glob('[!.]*'))
    with tqdm(total=len(logs), disable=True) as pbar:
        for summary_path in logs:
            try:
                if param_format == 'grid':
                    params = parse_grid_params_from_filename(summary_path.name)
                elif param_format == 'resurrection':
                    params = parse_resurrection_params_from_filename(summary_path.name)
                elif param_format == 'simplenet':
                    params = parse_simplenet_params_from_filename(summary_path.name)
                elif param_format == 'iclr':
                    params = parse_iclr_params_from_filename(summary_path.name)
                else:
                    raise ValueError(f'Unknown param format {param_format}')
                if config_dir.exists():
                    try:
                        config_params = parse_params_from_config(config_dir / summary_path.name)
                        # Check that stored params and extracted from filename agree
                        for param_k, param_v in params.items():
                            if param_k in config_params and param_v != config_params[param_k]:
                                raise RuntimeError(
                                    f'Corrupt config {summary_path.name} for {param_k}: parsed {param_v} '
                                    f'config {config_params[param_k]}')
                        # patch dataset if missing
                        if 'dataset' not in params and 'dataset' in config_params:
                            params['dataset'] = config_params['dataset']
                        has_config = True
                    except IndexError as ex:
                        print(f'Config not found for {summary_path.name} with exception {ex}')
                        has_config = False

                # patch dataset if missing
                elif 'dataset' not in params:
                    params['dataset'] = result_dir.name.replace('grid', '').replace('-', '').replace('_', '')
                    has_config = False
                else:
                    has_config = False

                has_log = validate_summary(summary_path)

                # compute parameter count
                if model_dir.exists():
                    try:
                        model_path = Path(model_dir / summary_path.name)
                        model_file = sorted(model_path.parent.glob(f'{model_path.name}*.tar'), key=os.path.getsize,
                                            reverse=True)
                        if len(model_file) == 1 and model_file[0].is_file():
                            model_file = model_file[0]
                        else:  # Its in a directory
                            model_file = sorted(model_path.glob('*.tar'), key=os.path.getsize, reverse=True)[0]
                        has_model = model_file.exists()
                        # if not has_model and purge:
                        #     shutil.rmtree(summary_path)

                    except IndexError as ex:
                        print(f'Model not found for {summary_path.name} with exception {ex}')
                        has_model = False
                        if purge:
                            print(f'Purging {summary_path.name}')
                            shutil.rmtree(summary_path)

                else:
                    has_model = False

                params['filename'] = summary_path.name
                index.append(params)
                data.append({'has_log': has_log, 'has_model': has_model, 'has_config': has_config})
            except ValueError as ex:
                print(ex)
                raise ex
            pbar.update()
            pbar.set_description(summary_path.name)
    if index and data:
        index = pd.DataFrame(index)
        table = pd.DataFrame(data, index=pd.MultiIndex.from_frame(index))
    else:
        table = pd.DataFrame(data)

    return table


def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
