import os
import shutil
import time
from pathlib import Path
from unittest import TestCase
import subprocess

from accelerate import notebook_launcher

from experiment.experiment import load_summary_table, load_event_file, load_model_from_checkpoint
from grid import main

os.environ['WANDB_MODE'] = 'offline'


class Test(TestCase):

    def test_log_callback_single_process(self):
        config_module = 'test/test/config/rectangular.py'

        results_dir = Path('results/test/single')
        os.environ['RESULTS_DIR'] = 'test/test'

        shutil.rmtree(results_dir, ignore_errors=True)
        # main(config_module)

        notebook_launcher(main, args=(config_module,), num_processes=1)
        summary_path = results_dir / 'logs/rect_2x2_an:relu_ar:norm_bs:128_la:0_lr:0.0001'
        event_files = sorted(summary_path.glob('*events*'), key=os.path.getsize, reverse=True)
        self.assertTrue(event_files[0].resolve().exists())
        summary = load_event_file(event_files[0].resolve())
        self.assertEqual(summary.shape[0], 24)

    def test_log_callback_single_multiprocess(self):
        config_module = 'test/test/config/single.py'

        results_dir = Path('results/test/single')
        os.environ['RESULTS_DIR'] = 'test/test'

        shutil.rmtree(results_dir, ignore_errors=True)
        # main(config_module)

        notebook_launcher(main, args=(config_module,), num_processes=4)
        summary_path = results_dir / 'logs/rect_2x2_an:relu_ar:norm_bs:128_la:0_lr:0.0001'
        event_files = sorted(summary_path.glob('*events*'), key=os.path.getsize, reverse=True)
        self.assertTrue(event_files[0].resolve().exists())
        summary = load_event_file(event_files[0].resolve())
        self.assertEqual(summary.shape[0], 24)

    def test_save_model_callback_single_process(self):
        config_module = 'test/test/config/single.py'

        results_dir = Path('results/test/single')
        os.environ['RESULTS_DIR'] = 'test/test'

        shutil.rmtree(results_dir, ignore_errors=True)
        # main(config_module)

        notebook_launcher(main, args=(config_module,), num_processes=1)
        model_path = results_dir / 'models/rect_2x2_an:relu_ar:norm_bs:128_la:0_lr:0.0001'
        model = load_model_from_checkpoint(model_path)
        self.assertEqual(str(model), """Rectangular(
  (Conv2d-0): Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (ReLU-0): ReLU()
  (Conv2d-1): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (ReLU-1): ReLU()
  (ConcatPool): Concatenate(layers=(AdaptiveAvgPool2d(output_size=(1, 1)), AdaptiveMaxPool2d(output_size=(1, 1))))
  (Flattening): Flatten(start_dim=1, end_dim=-1)
  (Linear-2): Linear(in_features=4, out_features=10, bias=True)
)""")

    def test_save_model_callback_multiprocess(self):
        config_module = 'test/test/config/single.py'

        results_dir = Path('results/test/single')
        os.environ['RESULTS_DIR'] = 'test/test'

        shutil.rmtree(results_dir, ignore_errors=True)
        # main(config_module)

        notebook_launcher(main, args=(config_module,), num_processes=4)
        model_path = results_dir / 'models/rect_2x2_an:relu_ar:norm_bs:128_la:0_lr:0.0001'
        model = load_model_from_checkpoint(model_path)
        self.assertEqual(str(model), """Rectangular(
  (Conv2d-0): Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (ReLU-0): ReLU()
  (Conv2d-1): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (ReLU-1): ReLU()
  (ConcatPool): Concatenate(layers=(AdaptiveAvgPool2d(output_size=(1, 1)), AdaptiveMaxPool2d(output_size=(1, 1))))
  (Flattening): Flatten(start_dim=1, end_dim=-1)
  (Linear-2): Linear(in_features=4, out_features=10, bias=True)
)""")
