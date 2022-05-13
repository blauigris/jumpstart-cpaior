from unittest import TestCase

import numpy as np
import torch
from torch import nn

from experiment.data import load_data
from experiment.train import set_seed
from jumpstart.loss import JumpstartRegularization
from jumpstart.util import compute_maxmin_preact_single, compute_maxmin_preact_average, compute_maxmin_preact_random
import os
os.environ['WANDB_MODE'] = 'offline'


class TestJumpstartRegularization(TestCase):

    def test_compute_maxmin(self):
        set_seed(42)
        trainloader, testloader, input_shape, output_shape = load_data(dataset='moons', random_state=123)
        X, y = next(iter(testloader))

        max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 0)
        max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 0)
        max_preact_random, min_preact_random = compute_maxmin_preact_random(X, 0)

        max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
            max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()
        max_preact_random, min_preact_random = max_preact_random.numpy(), min_preact_random.numpy()

        self.assertTrue(np.allclose(max_preact_average, max_preact_single))
        self.assertTrue(np.allclose(min_preact_average, min_preact_single))
        self.assertTrue(np.allclose(max_preact_random, max_preact_single))
        self.assertTrue(np.allclose(min_preact_random, min_preact_single))

        X, y = next(iter(testloader))
        max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 1)
        max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 1)
        max_preact_random, min_preact_random = compute_maxmin_preact_random(X, 1)

        max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
            max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()
        max_preact_random, min_preact_random = max_preact_random.numpy(), min_preact_random.numpy()

        self.assertTrue(np.allclose(max_preact_average, max_preact_single))
        self.assertTrue(np.allclose(min_preact_average, min_preact_single))
        self.assertTrue(np.allclose(max_preact_random, max_preact_single))
        self.assertTrue(np.allclose(min_preact_random, min_preact_single))

    def test_compute_maxmin_zero(self):
        with torch.no_grad():
            set_seed(42)
            X = torch.zeros((4, 3), dtype=torch.float, requires_grad=True)

            max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 0)
            max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 0)
            max_preact_random, min_preact_random = compute_maxmin_preact_random(X, 0)

            max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
                max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()
            max_preact_random, min_preact_random = max_preact_random.numpy(), min_preact_random.numpy()

            self.assertTrue(np.allclose(max_preact_average, max_preact_single))
            self.assertTrue(np.allclose(min_preact_average, min_preact_single))
            self.assertTrue(np.allclose(max_preact_random, max_preact_single))
            self.assertTrue(np.allclose(min_preact_random, min_preact_single))

            max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 1)
            max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 1)
            max_preact_random, min_preact_random = compute_maxmin_preact_random(X, 1)

            max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
                max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()
            max_preact_random, min_preact_random = max_preact_random.numpy(), min_preact_random.numpy()

            self.assertTrue(np.allclose(max_preact_average, max_preact_single))
            self.assertTrue(np.allclose(min_preact_average, min_preact_single))
            self.assertTrue(np.allclose(max_preact_random, max_preact_single))
            self.assertTrue(np.allclose(min_preact_random, min_preact_single))

    def test_compute_maxmin_conv(self):
        set_seed(42)
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10', random_state=123)
        X, y = next(iter(testloader))

        max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 0)
        max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 0)

        max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
            max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()

        self.assertTrue(np.allclose(max_preact_average, max_preact_single))
        self.assertTrue(np.allclose(min_preact_average, min_preact_single))

        X, y = next(iter(testloader))
        max_preact_average, min_preact_average = compute_maxmin_preact_average(X, 1)
        max_preact_single, min_preact_single = compute_maxmin_preact_single(X, 1)

        max_preact_average, min_preact_average, max_preact_single, min_preact_single = \
            max_preact_average.numpy(), min_preact_average.numpy(), max_preact_single.numpy(), min_preact_single.numpy()

        self.assertTrue(np.allclose(max_preact_average, max_preact_single))
        self.assertTrue(np.allclose(min_preact_average, min_preact_single))


    def test__get_constraint_for_axis_moons_average(self):
        set_seed(42)
        trainloader, testloader, input_shape, output_shape = load_data(dataset='moons', random_state=123)
        checkpoint = torch.load('./models/moons_untrained_test_model.pth')
        model = checkpoint['model']
        sepcons = JumpstartRegularization(model=model, aggr='mean', backprop_mode='average')
        X, y = next(iter(testloader))
        output = model(X)
        self.assertAlmostEqual(0.6894431710243225, float(sepcons.loss.detach().numpy()), places=4)
        self.assertEqual(50 * (4 + X.shape[0]) + X.shape[0] + 1,
                         sum([l.shape[0] for l in sepcons.parameter_losses.values()]))

    def test__get_constraint_for_axis_cifar(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10', random_state=123)
        # model = create_rectangular(50, 4, kind='relu', input_shape=input_shape, output_shape=output_shape)
        # torch.save({
        #     'model': model,
        # }, './cifar_untrained_test_model.pth')

        checkpoint = torch.load('./models/cifar_untrained_test_model.pth')
        model = checkpoint['model']
        sepcons = JumpstartRegularization(model=model, aggr='mean')
        X, y = next(iter(testloader))
        output = model(X)
        self.assertAlmostEqual(0.8274549841880798, float(sepcons.loss.detach().numpy()), places=4)
        self.assertEqual(50 * (4 + testloader.batch_size) + testloader.batch_size + 10,
                         sum([l.shape[0] for l in sepcons.parameter_losses.values()]))

    def test__get_constraint_for_axis_moons_single(self):
        set_seed(42)
        trainloader, testloader, input_shape, output_shape = load_data(dataset='moons', random_state=123)
        checkpoint = torch.load('./models/moons_untrained_test_model.pth')
        model = checkpoint['model']
        sepcons = JumpstartRegularization(model=model, aggr='mean', backprop_mode='single')
        X, y = next(iter(testloader))
        output = model(X)
        self.assertAlmostEqual(0.6894431710243225, float(sepcons.loss.detach().numpy()), places=4)
        self.assertEqual(50 * (4 + X.shape[0]) + X.shape[0] + 1,
                         sum([l.shape[0] for l in sepcons.parameter_losses.values()]))

    def test__get_constraint_for_axis_cifar_single(self):
        trainloader, testloader, input_shape, output_shape = load_data(dataset='cifar10', random_state=123)
        # model = create_rectangular(50, 4, kind='relu', input_shape=input_shape, output_shape=output_shape)
        # torch.save({
        #     'model': model,
        # }, './cifar_untrained_test_model.pth')

        checkpoint = torch.load('./models/cifar_untrained_test_model.pth')
        model = checkpoint['model']
        sepcons = JumpstartRegularization(model=model, aggr='mean', backprop_mode='single')
        X, y = next(iter(testloader))
        output = model(X)
        self.assertAlmostEqual(0.8274549841880798, float(sepcons.loss.detach().numpy()), places=4)
        self.assertEqual(50 * (4 + testloader.batch_size) + testloader.batch_size + 10,
                         sum([l.shape[0] for l in sepcons.parameter_losses.values()]))
