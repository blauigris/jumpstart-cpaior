from pathlib import Path
from unittest import TestCase

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from experiment.data import load_data
from experiment.experiment import load_summary_table, extract_grids
from experiment.grid import Rectangular
from experiment.plotting import plot_grids,\
    plot_decision_surface, plot_dead_units, plot_layer_sets_table, \
    export_grid_plots, plot_dead_points, plot_dead_units_points, plot_weights
from jumpstart import JumpstartMetrics


class Test(TestCase):





    def test_plot_dead_units_alive_moons(self):
        dataset = 'moons'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=8, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        model_state = torch.load('./moons_good_test_model.pth')
        model_state['Linear-50.weight'] = model_state['Linear-outputs.weight']
        model_state['Linear-50.bias'] = model_state['Linear-outputs.bias']
        del model_state['Linear-outputs.weight']
        del model_state['Linear-outputs.bias']
        model.load_state_dict(model_state)
        jumpstart_metrics = JumpstartMetrics(trainloader, model=model)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_dead_units(ax, jumpstart_metrics)
        plt.show()

    def test_plot_dead_points_moons_alive(self):
        dataset = 'moons'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=8, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        model_state = torch.load('./moons_good_test_model.pth')
        model_state['Linear-50.weight'] = model_state['Linear-outputs.weight']
        model_state['Linear-50.bias'] = model_state['Linear-outputs.bias']
        del model_state['Linear-outputs.weight']
        del model_state['Linear-outputs.bias']
        model.load_state_dict(model_state)
        jumpstart_metrics = JumpstartMetrics(trainloader, model=model)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_dead_points(ax, jumpstart_metrics)
        plt.show()

    def test_plot_dead_points_moons_dead(self):
        dataset = 'moons'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=8, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)
        jumpstart_metrics = JumpstartMetrics(trainloader, model=model)
        jumpstart_metrics.eval()
        print(jumpstart_metrics.dead_point_layer)

        fig, ax = plt.subplots(1, figsize=(20, 10))
        sns.despine()
        plot_dead_points(ax, jumpstart_metrics)
        plt.show()

    def test_plot_dead_units_points_moons_alive(self):
        dataset = 'moons'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=8, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        model_state = torch.load('./moons_good_test_model.pth')
        model_state['Linear-50.weight'] = model_state['Linear-outputs.weight']
        model_state['Linear-50.bias'] = model_state['Linear-outputs.bias']
        del model_state['Linear-outputs.weight']
        del model_state['Linear-outputs.bias']
        model.load_state_dict(model_state)
        jumpstart_metrics = JumpstartMetrics(trainloader, model=model)
        jumpstart_metrics.eval()

        fig = plt.figure()
        plot_dead_units_points(fig, jumpstart_metrics)
        plt.show()


    def test_plot_dead_units_dead_moons(self):
        dataset = 'moons'
        mode = 'partition'
        correction = None

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=8, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        jumpstart_metrics = JumpstartMetrics(trainloader, model=model)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_dead_units(ax, jumpstart_metrics)
        plt.show()

    def test_plot_dead_units_dead_cifar10(self):
        dataset = 'cifar10-toy'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=100, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        jumpstart_metrics = JumpstartMetrics(trainloader, model=model, compute_partitions=False)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_dead_units(ax, jumpstart_metrics)
        plt.show()

    def test_plot_layer_sets_table_moons(self):
        dataset = 'moons'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=100, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        jumpstart_metrics = JumpstartMetrics(trainloader, model=model, compute_partitions=False)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_layer_sets_table(ax, jumpstart_metrics)
        plt.show()

    def test_plot_layer_sets_table_cifar(self):
        dataset = 'cifar10-toy'

        # Load model
        trainloader, testloader, input_shape, output_shape = load_data(dataset=dataset, random_state=1,
                                                                       batch_size=100, n_samples=12)
        model = Rectangular(50, 4, activation='relu', input_shape=input_shape, output_shape=output_shape)

        jumpstart_metrics = JumpstartMetrics(trainloader, model=model, compute_partitions=False)
        jumpstart_metrics.eval()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        plot_layer_sets_table(ax, jumpstart_metrics)
        plt.show()
