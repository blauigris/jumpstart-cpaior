import os
import shutil
from pathlib import Path
from unittest import TestCase

import pandas as pd
from torchinfo import summary

from experiment.experiment import extract_metrics_from_summary, count_parameters_from_architecture, validate, \
    load_event_file
from experiment.grid import Rectangular
from grid import find_missing, main, load_config_table

os.environ['WANDB_MODE'] = 'offline'


class Test(TestCase):

    def test_extract_summary_nofile(self):
        filename = 'LEL'
        with self.assertRaises(ValueError):
            extract_metrics_from_summary(Path(filename))

    def test_count_parameters_from_architecture_dense(self):
        input_shape = (2,)
        output_shape = 1
        model = Rectangular(depth=30, width=45, input_shape=input_shape, output_shape=output_shape)
        expected = summary(model, verbose=0)
        computed = count_parameters_from_architecture(30, 45, kernel_size=None, input_shape=input_shape,
                                                      output_shape=output_shape)
        self.assertEqual(expected.trainable_params, computed['trainable_params'])

    def test_count_parameters_from_architecture_dense_2(self):
        input_shape = (2,)
        output_shape = 1
        model = Rectangular(depth=30, width=45, activation='relu', use_batchnorm=True, input_shape=input_shape,
                            output_shape=output_shape)
        expected = summary(model, verbose=0)
        computed = count_parameters_from_architecture(30, 45, kernel_size=None, input_shape=input_shape,
                                                      output_shape=output_shape, batchnorm=True)
        self.assertEqual(expected.trainable_params, computed['trainable_params'])
        self.assertEqual(expected.total_params, computed['total_params'])

    def test_count_parameters_from_architecture_conv(self):
        input_shape = (3, 32, 32)
        output_shape = 10
        model = Rectangular(depth=30, width=45, activation='relu', use_batchnorm=True, input_shape=input_shape,
                            output_shape=output_shape)
        expected = summary(model, verbose=1)
        computed = count_parameters_from_architecture(30, 45, kernel_size=(3, 3), input_shape=input_shape,
                                                      output_shape=output_shape, batchnorm=True)
        self.assertEqual(expected.trainable_params, computed['trainable_params'])
        self.assertEqual(expected.total_params, computed['total_params'])

    def test_validate_2(self):
        result_dir = '../results/imagewoof160'
        table = validate(Path(result_dir))

        expected = (8, 3)
        self.assertTrue(table['has_log'].all())
        self.assertFalse(table['has_model'].all())
        self.assertEqual(table.shape, expected)

    def test_restart_from_config_missing(self):
        config_module = 'test/test/config/rectangular.py'
        os.environ['RESULTS_DIR'] = 'test/results'
        results_dir = Path('results/rectangular')
        shutil.rmtree(results_dir, ignore_errors=True)
        main(config_module)
        config = load_config_table(config_module)
        found = validate(results_dir)
        missing = find_missing(config, found)
        self.assertEqual(len(missing), 0)

        found = validate(results_dir)
        bad = found.reset_index().sample(n=3, replace=False)
        for bad_filename in bad['filename']:
            shutil.rmtree(results_dir / 'models' / bad_filename)

        found = validate(results_dir)
        missing = find_missing(config, found).reset_index()[missing.columns.to_list()]
        bad = bad[missing.columns.to_list()]
        missing_missing = pd.concat([bad, missing]).drop_duplicates(keep=False)
        self.assertEqual(missing_missing.shape[0], 0)

        main(config_module)
        found = validate(results_dir)
        missing = find_missing(config, found)
        shutil.rmtree(results_dir, ignore_errors=True)
        self.assertEqual(len(missing), 0)

    def test_load_event_file(self):
        with self.assertRaises(Exception):
            load_event_file('patata')

    def test_cifar10(self):
        config_module = 'test/test/config/cifar10.py'
        os.environ['RESULTS_DIR'] = 'test/results'
        results_dir = Path('results/cifar10')
        shutil.rmtree(results_dir, ignore_errors=True)
        main(config_module)
        config = load_config_table(config_module)
        found = validate(results_dir)
        missing = find_missing(config, found)
        self.assertEqual(len(missing), 0)

    def test_cifar100(self):
        config_module = 'test/test/config/cifar100.py'
        os.environ['RESULTS_DIR'] = 'test/results'
        results_dir = Path('results/cifar100')
        shutil.rmtree(results_dir, ignore_errors=True)
        main(config_module)
        config = load_config_table(config_module)
        found = validate(results_dir)
        missing = find_missing(config, found)
        self.assertEqual(len(missing), 0)


