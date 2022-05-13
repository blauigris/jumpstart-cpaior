import torch
import torch.nn.functional as F

from jumpstart.util import compute_maxmin_preact_average, compute_maxmin_preact_single, compute_maxmin_preact_random


class JumpstartRegularization:
    def __init__(self, model=None, balance=None, axis=(0, 1), aggr='norm', backprop_mode='single', negative_margin=-1):
        self._negative_margin = None
        self.negative_margin = negative_margin
        self.aggr = aggr
        self.axis = axis  # 0 -> unit, 1 -> point
        self.balance = balance if balance is not None else 0.5
        self.parameter_losses = {}
        self.hook_handlers = {}
        self._model = None
        self.backprop_mode = backprop_mode

        if self.backprop_mode == 'average':
            self.compute_maxmin_preact = compute_maxmin_preact_average
        elif self.backprop_mode == 'single':
            self.compute_maxmin_preact = compute_maxmin_preact_single
        elif self.backprop_mode == 'random':
            self.compute_maxmin_preact = compute_maxmin_preact_random
        else:
            raise ValueError(f'Unknown backprop mode {self.backprop_mode}')

        self.model = model


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        # Avoid setting the model twice
        if model is not self.model:
            self._model = model
            print(f'Setting Sepcons hooks with {self.aggr} aggregation and {self.backprop_mode} backprop mode')
            self.parameter_losses.clear()
            for name, module in self.model.named_modules():
                if hasattr(module, 'weight') and not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    print(f'Registering layer {name} of type {type(module).__name__}')
                    handler = module.register_forward_hook(self.hook)
                    self.hook_handlers[name] = handler

    def hook(self, module, input_, output):
        for ax in self.axis:
            cons = self._get_constraint_for_axis(output, ax)
            self.parameter_losses[(module, ax)] = cons

    def _get_constraint_for_axis(self, x, ax):
        max_preact, min_preact = self.compute_maxmin_preact(x, ax)

        constraint = self.balance * F.relu(1 - max_preact) + (1 - self.balance) * F.relu(self._negative_margin + min_preact)

        return constraint

    @property
    def loss(self):
        cat = torch.cat(list(self.parameter_losses.values()))
        if self.aggr == 'mean':
            res = torch.mean(cat)
        elif self.aggr == 'norm':
            res = torch.norm(cat)
        elif self.aggr == 'sum':
            res = torch.sum(cat)
        else:
            raise ValueError(f'Unknown loss mode {self.aggr}')

        return res

    @property
    def negative_margin(self):
        return -self._negative_margin

    @negative_margin.setter
    def negative_margin(self, value):
        self._negative_margin = -value
