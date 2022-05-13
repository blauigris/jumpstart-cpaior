from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.nn import Sequential
from torch.utils.data import RandomSampler

from jumpstart.util import compute_maxmin_preact_single as compute_maxmin_preact


class JumpstartMetrics:
    def __init__(self, dataloader=None, model=None, device=None, affine_flag_value=1, dead_flag_value=-1,
                 intersection_flag_value=0, compute_partitions=False, batch_size=None, enabled=False):
        self.compute_partitions = compute_partitions
        self.intersection_flag_value = intersection_flag_value
        self.dead_flag_value = dead_flag_value
        self.affine_flag_value = affine_flag_value
        if dataloader:
            if isinstance(dataloader.sampler, RandomSampler):
                print('WARNING: Dataloader with shuffle=True, replacing it')
                dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size,
                                                         shuffle=False, num_workers=dataloader.num_workers)
            self.batch_size = dataloader.batch_size
        else:
            self.batch_size = batch_size

        self.dataloader = dataloader
        self.n_layers = 0
        self._model = None
        self.model = model
        if device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = device
        self.current_batch = None
        self._jumpstart_metrics_eval_mode = False
        self.n_units = None
        self.n_points = None
        self.n_dead_points = None
        self.n_affine_units = None
        self.n_separating_units = None
        self.n_dead_units = None
        self.n_dead_points = None
        self.dead_units = {}
        self.affine_units = {}
        self._output = defaultdict(list)
        self.output = None
        self.layer_to_partitions = {}
        self.layer_sets = {}
        self.enabled = enabled

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        # Avoid setting the model twice
        if model is not self.model:
            self.n_layers = 0
            self._model = model
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight'):
                    self.n_layers += 1
                    module.register_forward_hook(partial(self.hook, name=name))

    def reset(self):
        self.dead_units.clear()
        self._output.clear()
        self.layer_to_partitions.clear()
        self.layer_sets.clear()
        self.n_dead_points = 0
        self.n_dead_units = 0
        self.n_affine_units = 0
        self.n_separating_units = 0
        self.n_units = 0
        self.n_points = 0
        self.current_batch = 0

    def update(self):
        for affine_units in self.affine_units.values():
            self.n_affine_units += int(affine_units.sum())
        for dead_units in self.dead_units.values():
            self.n_dead_units += int(dead_units.sum())
            self.n_units += dead_units.shape[0]
        self.n_separating_units = self.n_units - (self.n_dead_units + self.n_affine_units)
        self.n_dead_points = len(self.dead_points)
        self.n_points = len(set().union(*[layer_set for layer_set in next(iter(self.layer_sets.values())).values()]))

        if self.compute_partitions:
            self.output = {name: np.concatenate(batch_outputs) for name, batch_outputs in self._output.items()}
            for name, layer_output in self.output.items():
                layer_partitions = []
                layer_output = layer_output.T if layer_output.ndim == 2 else layer_output.transpose(1, 0, 2, 3)
                for unit_output in layer_output:
                    # TODO store pixel permutations separately for further analysis
                    unit_output = unit_output.flatten()
                    partition = compute_partition(unit_output)
                    layer_partitions.append(partition)

                self.layer_to_partitions[name] = tuple(layer_partitions)

    def eval(self):
        # Set compute mode to avoid capturing the regular training batches
        self.reset()
        self.enabled = True
        self.model.to(self.device)
        with torch.no_grad():
            for i, (inputs, _) in enumerate(self.dataloader):
                self.current_batch = i
                inputs = inputs.to(self.device)
                self.model(inputs)
        self.enabled = False
        self.update()

    def hook(self, module, input, output, name=None):
        if self.enabled:
            output = output.detach().cpu()
            self._update_point_layer_metrics(output, name=name)
            self._update_unit_metrics(output, name=name)
            if self.compute_partitions:
                self._update_outputs(output, name=name)

    def _update_outputs(self, output, name):
        self._output[name].append(output)

    def _update_point_layer_metrics(self, output, name=None):
        max_preact, min_preact = compute_maxmin_preact(output, ax=1)

        zero_set = torch.nonzero(max_preact <= 0, as_tuple=False).numpy().flatten()
        zero_set += self.current_batch * self.batch_size
        affine_set = torch.nonzero(min_preact > 0, as_tuple=False).numpy().flatten()
        affine_set += self.current_batch * self.batch_size
        batch = np.arange(self.current_batch * self.batch_size,
                          self.current_batch * self.batch_size + output.shape[0])
        intersection_set = np.setdiff1d(np.setdiff1d(batch, zero_set), affine_set)

        layer_sets = {'zero': set(zero_set), 'affine': set(affine_set), 'intersection': set(intersection_set)}
        if name in self.layer_sets:
            for set_name, set_points in layer_sets.items():
                self.layer_sets[name][set_name].update(set_points)
        else:
            self.layer_sets[name] = layer_sets

    def _update_unit_metrics(self, output, name=None):
        max_preact, min_preact = compute_maxmin_preact(output, ax=0)

        dead_units = max_preact <= 0
        if name in self.dead_units:
            self.dead_units[name] = np.logical_and(self.dead_units[name], dead_units)
        else:
            self.dead_units[name] = dead_units

        affine_units = min_preact > 0
        if name in self.affine_units:
            self.affine_units[name] = np.logical_and(self.affine_units[name], affine_units)
        else:
            self.affine_units[name] = affine_units

    @property
    def dead_unit_table(self):
        return pd.DataFrame.from_dict(self.dead_units, orient='index').astype(float).astype('boolean').T

    @property
    def affine_unit_table(self):
        return pd.DataFrame.from_dict(self.affine_units, orient='index').astype(float).astype('boolean').T

    @property
    def all_unit_table(self):
        units = self.affine_unit_table * self.affine_flag_value + self.dead_unit_table * self.dead_flag_value
        return units

    @property
    def dead_points(self):
        return set().union(*[layer_set['zero'] for layer_set in self.layer_sets.values()])

    @property
    def dead_point_layer(self):
        dead_point_layer = {}
        for layer_name, layer_sets in self.layer_sets.items():
            for dead_point in layer_sets['zero']:
                if dead_point not in dead_point_layer:
                    dead_point_layer[dead_point] = layer_name
        return dead_point_layer


    @property
    def layer_sets_table(self):
        data = {}
        dead_points, intersection_points = set(), set()
        for layer_name, layer_sets in self.layer_sets.items():
            points = np.zeros(self.n_points) * np.nan
            dead_points = dead_points.union(layer_sets['zero'])
            intersection_points = intersection_points.union(layer_sets['intersection']) - dead_points
            affine_points = layer_sets['affine'] - dead_points - intersection_points
            points[list(dead_points)] = self.dead_flag_value
            points[list(affine_points)] = self.affine_flag_value
            points[list(intersection_points)] = self.intersection_flag_value
            data[layer_name] = points

        table = pd.DataFrame(data)
        return table

    @property
    def data_jumpstart(self):
        return {unit for layer in self.layer_to_partitions.values() for unit in layer}

    @property
    def output_table(self):
        data = {k: pd.DataFrame(v.T).stack().T for k, v in self.output.items()}
        return pd.DataFrame(data)

    @property
    def metrics(self):
        return {
            'n_dead_points': self.n_dead_points,
            'dead_point_ratio': self.n_dead_points / self.n_points,
            'n_dead_units': self.n_dead_units,
            'dead_unit_ratio': self.n_dead_units / self.n_units,
            'n_affine_units': self.n_affine_units,
            'affine_unit_ratio': self.n_affine_units / self.n_units,
            'n_separating_units': self.n_separating_units,
            'separating_unit_ratio': self.n_separating_units / self.n_units,
            'n_units': self.n_units,
            'n_points': self.n_points,
        }

