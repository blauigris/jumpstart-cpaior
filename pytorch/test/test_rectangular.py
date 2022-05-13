import logging
import sys
from unittest import TestCase

from torch import nn

from experiment.data import load_data
from experiment.grid import Rectangular
from torchsummary import summary
import numpy as np

logger = logging.getLogger('TEST_RECTANGULAR')
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

import os
os.environ['WANDB_MODE'] = 'offline'

class Test(TestCase):
    def test_create_rectangular_maxpool_log(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10',
                                                                       batch_size=128)
        use_maxpool = 3
        maxpool_mode = 'log'
        model = Rectangular(16, 30, input_shape=input_shape, output_shape=output_shape,
                            n_maxpool=use_maxpool, maxpool_mode=maxpool_mode)

        expected_poolings = [8, 12, 14]
        for i, exp_mp in enumerate(expected_poolings):
            self.assertIsInstance(model[exp_mp * 2 + i], nn.MaxPool2d)

    def test_create_rectangular_residual(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10',
                                                                       batch_size=128)
        model = Rectangular(16, 30, input_shape=input_shape, output_shape=output_shape,
                            skip_connections=2)

        expected_residual = np.arange(2, 50, 2)
        model(next(iter(trainloader))[0])

    def test_create_rectangular_maxpool_log_2(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10',
                                                                       batch_size=128)
        n_maxpool = 3
        maxpool_mode = 'log'
        model = Rectangular(30, 30, input_shape=input_shape, output_shape=output_shape,
                            n_maxpool=n_maxpool, maxpool_mode=maxpool_mode)

        expected_poolings = [15, 23, 27]
        for i, exp_mp in enumerate(expected_poolings):
            self.assertIsInstance(model[exp_mp * 2 + i], nn.MaxPool2d)


    def test_create_rectangular_no_maxpool(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10',
                                                                       batch_size=128)
        n_maxpool = 0
        maxpool_mode = 'log'
        model = Rectangular(10, 64, input_shape=input_shape, output_shape=output_shape,
                            n_maxpool=n_maxpool, maxpool_mode=maxpool_mode)
        summary(model, input_size=input_shape)
        for layer in model:
            print(layer)
            self.assertFalse(isinstance(layer, nn.MaxPool2d))