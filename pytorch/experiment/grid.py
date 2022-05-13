import json
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import init
from torchvision.models.resnet import Bottleneck, ResNet
from tqdm import tqdm

# from experiment.callbacks import TrainingCallback, SaveModel, NotificationCallback, send_notification
from experiment.callbacks import TrainingCallback, SaveModel, AnnealingDropoutCallback
from experiment.data import load_data
from experiment.train import train


def create_resnet(depth, width, input_shape, output_shape, dropout_rate=0, init='kaiming'):
    if dropout_rate > 0:
        raise NotImplementedError(f'Dropout not implemented')
    if init != 'kaiming':
        raise NotImplementedError('Only kaiming init implemented')
    if input_shape[0] != 3:
        raise ValueError('ResNet only works with three channels')

    if depth == 50:
        block = Bottleneck
        layers = [3, 4, 6, 3]

    else:
        raise ValueError(f'Unsupported depth {depth}')

    model = ResNet(block, layers, width_per_group=width, num_classes=output_shape)

    return model


class Concatenate(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, data):
        outputs = []
        for layer in self.layers:
            outputs.append(layer(data))
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by accelerator.printing an object of this class.
        return f'layers=({", ".join([repr(layer) for layer in self.layers])})'


class Rectangular(nn.Sequential):
    def __init__(self, depth, width, input_shape, output_shape, activation='relu', kernel=3, use_flattening=False,
                 use_batchnorm=False, dropout_rate=False, n_maxpool=None, maxpool_mode='log', skip_connections=None,
                 init='kaiming'):
        self._dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.use_flattening = use_flattening
        self.output_shape = output_shape
        self.skip_connections = skip_connections
        self.maxpool_mode = maxpool_mode
        self.n_maxpool = n_maxpool
        self.kernel = kernel
        self.activation = activation
        self.input_shape = input_shape
        self.width = width
        self.depth = depth
        self.init = init
        layers = self.create_layers()
        layers = self.add_top(layers)
        super().__init__(layers)

        if self.init != 'kaiming':
            self.initialize()

    def initialize(self):
        for layer in self:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if self.init == 'kaiming':
                    layer.reset_parameters()
                elif self.init == 'zero':
                    init.zeros_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)
                elif self.init == 'glorot':
                    torch.nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
                        bound = 1 / math.sqrt(fan_in)
                        init.uniform_(self.bias, -bound, bound)
                else:
                    raise ValueError(f'Unknown init scheme {self.init}')

    def add_top(self, layers):
        if len(self.input_shape) > 1:  # conv
            if self.use_flattening:
                features = nn.Sequential(layers)
                if torch.cuda.is_available():
                    dtype = torch.cuda.FloatTensor
                else:
                    dtype = torch.FloatTensor
                dummy_x = torch.rand(2, *self.input_shape).type(dtype)
                outputs = features(dummy_x)
                flattened_input_shape = torch.prod(torch.tensor(outputs.shape[1:], requires_grad=False))
                layers['Flattening'] = nn.Flatten(1, -1)
                layers[f'Linear-{self.depth}'] = nn.Linear(flattened_input_shape, self.output_shape)
            else:
                global_pool = Concatenate([nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.AdaptiveMaxPool2d((1, 1))])

                # global_pool = nn.AdaptiveAvgPool2d((1, 1))
                layers[f'ConcatPool'] = global_pool
                layers['Flattening'] = nn.Flatten(1, -1)
                layers[f'Linear-{self.depth}'] = nn.Linear(self.width * 2, self.output_shape)
        else:  # dense
            layers[f'Linear-{self.depth}'] = nn.Linear(self.width, self.output_shape)

        return layers

    def create_layers(self):
        if self.activation not in {'relu'}:
            raise ValueError(f'Unknown activation {self.activation}')
        if len(self.input_shape) > 1 and self.n_maxpool:
            if self.maxpool_mode == 'log':
                maxpool_layers = [self.depth // 2]
                for _ in range(self.n_maxpool - 1):
                    maxpool_layers.append(maxpool_layers[-1] // 2)

                maxpool_layers = self.depth - np.array(maxpool_layers)
                # Adjust position due to the inclusion of the previous maxpool layers
                maxpool_layers += np.arange(len(maxpool_layers))
            elif self.maxpool_mode == 'linear':
                maxpool_layers = np.linspace(0, self.depth, num=self.n_maxpool)
            else:
                raise ValueError(f'Unknown mode {self.maxpool_mode}')
        else:
            maxpool_layers = []

        layers = OrderedDict()
        for d in range(self.depth + len(maxpool_layers)):
            if d in maxpool_layers:
                layers[f'MaxPool2d-{d}'] = nn.MaxPool2d(2)
            else:
                if len(self.input_shape) > 1:
                    layer = nn.Conv2d(self.width if d > 0 else self.input_shape[0], self.width, self.kernel, padding=1)
                    layers[f'Conv2d-{d}'] = layer
                else:
                    layer = nn.Linear(self.width if d > 0 else self.input_shape[0], self.width)
                    layers[f'Linear-{d}'] = layer

                if self.use_batchnorm:
                    if len(self.input_shape) > 1:
                        layers[f'BN-{d}'] = (nn.BatchNorm2d(self.width))
                    else:
                        layers[f'BN-{d}'] = nn.BatchNorm1d(self.width)
                if self.dropout_rate:
                    layers[f'Dropout-{d}'] = nn.Dropout(self.dropout_rate)

                layers[f'ReLU-{d}'] = nn.ReLU()

        return layers

    def forward(self, input_):
        relu_count = 0
        for module in self:
            # for name, module in list(self.named_modules())[1:]:

            input_ = module(input_)
            if self.skip_connections:
                if isinstance(module, nn.ReLU):
                    if relu_count == 0:
                        residual = input_
                    if relu_count % self.skip_connections == 0:
                        input_ = input_ + residual
                        residual = input_

                    relu_count += 1

        return input_

    @property
    def dropout_rate(self):
        return self._dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value
        dropout_layers = 0
        for module in self:
            # for name, module in list(self.named_modules())[1:]:
            if isinstance(module, nn.Dropout):
                dropout_layers += 1
                module.p = self.dropout_rate

        if dropout_layers == 0:
            raise ValueError('Setting dropout rate but no dropout layers found')


def run_grid(param_grid, dataset, epochs, *, project=None, model_dir=None, log_dir=None, num_workers=0,
             verbose=0, config_dir=None, n_samples=100, notify=2, grid_name=None, seed=None, split_batches=True):
    # Create accelerator just for printing
    accelerator = Accelerator(split_batches=split_batches)

    accelerator.print(f'Training for {epochs} epochs on {dataset} ')
    accelerator.print(f'Storing logs at {log_dir}, models at {model_dir} and config at {config_dir}')
    param_keys = set(param_grid.columns)
    default_params = {'lambda', 'aggr', 'use_flattening', 'n_maxpool', 'maxpool_mode', 'n_samples'} - param_keys
    default_warning = f'WARNING: using default on the following params {default_params}'
    accelerator.print(default_warning)
    grid_name = grid_name if grid_name else ' '.join(param_keys)

    with tqdm(total=len(param_grid)) as pbar:
        for _, params in param_grid.iterrows():
            params = params.to_dict()
            hparams = params.copy()
            # Set defaults
            model_type = params.pop('model_type', 'rect')
            if model_type == 'resnet':
                params['use_batchnorm'] = True
                params['n_maxpool'] = 'resnet'
                params['maxpool_mode'] = 'resnet'

            depth = params.pop('depth', None)
            width = params.pop('width', None)
            # Update filename
            filename = f'{model_type}_{depth}x{width}_' + encode_params_str(params)
            pbar.set_description(filename)

            lr = params.pop('lr')
            batch_size = params.pop('batch_size')

            # More defaults
            aggr = params.pop('aggr', 'norm')
            lambda_ = params.pop('lambda', 0)
            use_flattening = params.pop('use_flattening', False)
            use_batchnorm = params.pop('use_batchnorm', False)
            n_maxpool = params.pop('n_maxpool', 0)
            maxpool_mode = params.pop('maxpool_mode', None)
            n_samples = params.pop('n_samples', 100)
            noise = params.pop('n_samples', 0)
            scheduler = params.pop('scheduler', None)
            no_xent = params.pop('no_xent', False)
            axis = params.pop('axis', (0, 1))
            activation = params.pop('activation', 'relu')
            backprop_mode = params.pop('backprop_mode', 'single')
            balance = params.pop('balance', 0.50)
            annealing_dropout_epochs = params.pop('annealing_dropout_epochs', 0)
            dropout_rate = params.pop('dropout_rate', 0)
            init = params.pop('init', 'kaiming')

            if len(params) > 0:
                raise ValueError(f'Unused params {params}')

            callbacks = []

            if log_dir:
                callbacks.append(TrainingCallback(
                    project=project,
                    config=hparams,
                    log_dir=Path(log_dir) / filename))
            if model_dir:
                callbacks.append(SaveModel(Path(model_dir) / filename))
            if config_dir:
                store_config(locals(), config_dir, filename)
            try:
                if annealing_dropout_epochs:
                    callbacks.append(AnnealingDropoutCallback(annealing_dropout_epochs, dropout_rate))

                trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, n_samples=n_samples,
                                                                               batch_size=batch_size,
                                                                               num_workers=num_workers,
                                                                               noise=noise,
                                                                               random_state=seed)
                if model_type == 'rect':
                    model = Rectangular(depth, width, activation=activation, input_shape=input_shape,
                                        output_shape=output_shape, use_batchnorm=use_batchnorm,
                                        use_flattening=use_flattening, n_maxpool=n_maxpool, maxpool_mode=maxpool_mode,
                                        dropout_rate=dropout_rate, init=init)
                elif model_type == 'resnet':
                    model = create_resnet(depth, width, input_shape=input_shape, output_shape=output_shape,
                                          dropout_rate=dropout_rate, init=init)
                else:
                    raise ValueError(f'Unknown model type {model_type}')


                train(model=model, trainloader=trainloader, testloader=testloader, input_shape=input_shape,
                      output_shape=output_shape, epochs=epochs, lr=lr, lambda_=lambda_, no_xent=no_xent,
                      aggr=aggr, callbacks=callbacks, axis=axis, scheduler=scheduler, balance=balance,
                      backprop_mode=backprop_mode, split_batches=split_batches)

            except Exception as ex:
                accelerator.print(f'FAILED {params} from {grid_name} due to {ex}')
                raise ex
            pbar.update(1)



def store_config(config, config_dir, filename):
    timestamp = '{:%d-%m-%Y-%H-%M-%S}'.format(datetime.now())
    filepath = config_dir / f'{filename}-{timestamp}.json'
    config = {str(c): str(v) for c, v in config.items()}
    with open(filepath, 'w') as f:
        json.dump(config, f)


def encode_params_str(params):
    params_str = []
    for k, v in params.items():
        letters = []
        words = [w for w in k.split('_') if len(w)]
        if len(words) > 1:
            for word in words:
                letters.append(word[0])
        else:
            letters.extend([words[0][0], words[0][-1]])

        letters = ''.join(letters)
        param_str = f'{letters}:{v}'
        params_str.append(param_str)
    return '_'.join(params_str)
