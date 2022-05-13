import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from jumpstart.metrics import JumpstartMetrics


class Callback:
    def __init__(self): pass

    def on_train_begin(self, logs=None): pass

    def on_train_end(self, logs=None): pass

    def on_epoch_begin(self, logs=None): pass

    def on_epoch_end(self, logs=None): pass

    def on_batch_begin(self, logs=None): pass

    def on_batch_end(self, logs=None): pass

    def on_loss_begin(self, logs=None): pass

    def on_loss_end(self, logs=None): pass

    def on_step_begin(self, logs=None): pass

    def on_step_end(self, logs=None): pass


class StopTrainingCallback(Callback):

    def on_epoch_end(self, logs=None):
        if logs['metric_data']['val_acc'] == 1:
            raise StopIteration('End of training due to val_acc == 1')


class TrainingCallback(Callback):

    def __init__(self, project, config, log_dir=None, compute_jumpstart_metrics=False, plot_freq=1, plot=(), plot_dir=None,
                 dataloader=None, compute_partitions=False):
        super().__init__()
        self.config = config
        self.project = project
        self.compute_partitions = compute_partitions
        self.plot_dir = plot_dir
        self.dataloader = dataloader
        self.plot = plot
        if self.plot_dir:
            self.plot_dir = Path(self.plot_dir)
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.compute_jumpstart_metrics = compute_jumpstart_metrics
        self.jumpstart_metrics = None
        self.plot_freq = plot_freq
        self.log_dir = log_dir
        self.writer = None
        self.run = None

    def on_train_begin(self, logs=None):
        # Store summaries in a directory inside the experiment directory
        self.run = wandb.init(project=self.project, config=self.config)

        if self.log_dir:
            self.writer = SummaryWriter(self.log_dir)
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics = JumpstartMetrics(model=logs['model'], batch_size=logs['params']['batch_size'],
                                                      enabled=False, compute_partitions=self.compute_partitions)

    def on_train_end(self, logs=None):
        self.run.finish()
        if self.log_dir:
            self.writer.flush()
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics.enabled = False

    def on_batch_begin(self, logs=None):
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics.enabled = True  # Capture training activations

    def on_batch_end(self, logs=None):
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics.current_batch = logs['batch']
            self.jumpstart_metrics.enabled = False  # Stop capturing validation stuff

    def on_epoch_begin(self, logs=None):
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics.reset()

    def on_epoch_end(self, logs=None):
        # Store metric_data
        metrics = logs['metric_data']
        epoch = logs['epoch']
        if self.compute_jumpstart_metrics:
            self.jumpstart_metrics.update()
        if self.log_dir:
            for key, val in metrics.items():
                self.writer.add_scalar(key, val, epoch)
            if self.compute_jumpstart_metrics:
                for key, val in self.jumpstart_metrics.metrics.items():
                    self.writer.add_scalar(key, val, epoch)

        self.run.log(metrics, epoch)


def get_metrics_str(metrics):
    metrics_str = []
    for key, value in metrics.items():
        try:
            metrics_str.append(f'{key}: {value:.6}')
        except ValueError:
            metrics_str.append(f'{key}: {value}')

    return ' '.join(metrics_str)


class STDOutLoggingCallback(Callback):
    def __init__(self, logger, epoch_freq=1, batch_freq=float('inf')):
        self.batch_freq = batch_freq
        self.epoch_freq = epoch_freq
        self.logger = logger

    # def on_batch_end(self, logs):
    #     batch = logs['batch']
    #     if not batch % self.batch_freq:
    #         epoch = logs['epoch']
    #         self.logger.debug(
    #             f'[Epoch {epoch} Batch {batch}] ' + ' '.join(
    #                 f'{key}: {value}' for key, value in logs['metric_data'].items()))

    def on_epoch_end(self, logs):
        epoch = logs['epoch']
        if not epoch % self.epoch_freq:
            metrics_str = get_metrics_str(logs['metric_data'])
            self.logger.info(f'[Epoch {epoch}] ' + metrics_str)


class SaveModel(Callback):

    def __init__(self, output_dir, filename=''):
        super().__init__()
        self.filename = filename
        self.output_dir = Path(output_dir)
        self.timestamp = '{:%d-%m-%Y-%H-%M-%S}'.format(datetime.datetime.now())

    # def on_train_begin(self, logs=None):
    #     self.output_dir.mkdir(parents=True, exist_ok=True)
    #     model = logs['model']
    #     optimizer = logs['optimizer']
    #     metrics = logs['metric_data']
    #     filename_str = f'INIT-{self.timestamp}.tar'
    #     if self.filename:
    #         filename_str = f'{self.filename}-{filename_str}'
    #     path = self.output_dir / filename_str
    #     torch.save({
    #         'epoch': 'INIT',
    #         'model': model,
    #         'optimizer': optimizer,
    #         'metric_data': metrics,
    #     }, path)

    def on_train_end(self, logs=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epoch = logs['epoch']
        model = logs['model']
        optimizer = logs['optimizer']
        metrics = logs['metric_data']
        filename_str = f'{epoch}-{self.timestamp}.tar'
        if self.filename:
            filename_str = f'{self.filename}-{filename_str}'
        path = self.output_dir / filename_str
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'metric_data': metrics,
        }, path)


class AnnealingDropoutCallback(Callback):
    def __init__(self, epochs, rate=1.0):
        super().__init__()
        self.rate = rate
        self.epochs = epochs

    def on_epoch_end(self, logs=None):
        epoch = logs['epoch']
        v = max(0, self.rate - self.rate * epoch / self.epochs)
        # print("epoch", epoch, "update", v)
        logs['model'].dropout_rate = v
