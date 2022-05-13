import datetime
import os
import random as rn
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN
from keras.layers import Activation
from slugify import slugify

from callbacks import ActivationPlotNetwork, AnnealingDropoutCallback
from callbacks import SurfaceCallback, NanAlert, MatrixPlotNetwork
from datasets.datasets import load_dataset
from lr_utils.keras_CLR import CyclicLR, CLR
from lr_utils.keras_SGDR import SGDRScheduler
from lr_utils.keras_lr_finder import LRFinder
from util import binary_hinge_accuracy


def get_callbacks(histogram_freq, write_grads, write_graph, write_images, epoch_freq, batch_freq, x_train,
                  y_train, epoch_start, show_on_train, show_input_layer, show_local_layers,
                  check_units, check_numerics, check_weights, check_optimizer, check_gradients, filename, store_img,
                  embeddings_freq,
                  embeddings_layer_names,
                  plot_matrix
                  , log_dir='summaries', show_decision_layers=False, plot_activations=False, plot_surface=False,
                  single_plot=False, annealing_dropout_epochs=None, annealing_dropout_rate=0.5,
                  print_weights=False, plot_epsilon=0.01):
    log_dir = os.path.join(log_dir, filename)

    callbacks = [TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq, write_grads=write_grads,
                             write_graph=write_graph, write_images=write_images, embeddings_freq=embeddings_freq,
                             embeddings_layer_names=embeddings_layer_names), TerminateOnNaN()]

    param_list = filename.split('-')
    title = ' '.join(param_list[:6] + ['\n'] + param_list[6:])
    if plot_surface:
        surface = SurfaceCallback(x_train, y_train, title=title,
                                  epoch_start=epoch_start, epoch_freq=epoch_freq,
                                  batch_freq=batch_freq,
                                  show_on_train=show_on_train,
                                  show_input_layer=show_input_layer,
                                  show_local_layers=show_local_layers,
                                  show_decision_layers=show_decision_layers,
                                  store_img=store_img,
                                  check_units=check_units,
                                  single_plot=single_plot,
                                  print_weights=print_weights,
                                  epsilon=plot_epsilon,
                                  )
        callbacks.append(surface)

    if plot_matrix:
        matrix = MatrixPlotNetwork(batch_freq=batch_freq, epoch_freq=epoch_freq, title=title)
        callbacks.append(matrix)

    if plot_activations:
        activations = ActivationPlotNetwork(batch_freq=batch_freq, epoch_freq=epoch_freq, title=title, X=x_train,
                                            store_img=store_img)
        callbacks.append(activations)

    if check_numerics or check_units or check_weights or check_optimizer or check_gradients:
        nan_alert = NanAlert(x_train, y_train,
                             check_weights=check_weights,
                             check_gradients=check_gradients,
                             check_units=check_units,
                             check_optimizer=check_optimizer,
                             check_numerics=check_numerics
                             )
        callbacks.append(nan_alert)

    if annealing_dropout_epochs:
        annealing_dropout_callback = AnnealingDropoutCallback(epochs=annealing_dropout_epochs,
                                                              rate=annealing_dropout_rate)
        callbacks.append(annealing_dropout_callback)

    return callbacks


def params2str(name, loss, optimizer, lr, batch_size, dataset, network_parameters, extra):
    loss_str = str(loss)
    param_list = [name] if name else [dataset]
    param_list += get_param_str(network_parameters) + \
                  ['{}:{}:{}-bs-{}'.format(loss_str, optimizer, lr, batch_size),
                   ]
    if extra:
        param_list.append(extra)

    return param_list


def get_param_str(network_parameters):
    str_params = []
    if network_parameters is not None:
        netparams = network_parameters.copy()
        if 'activation' in netparams:
            str_params.append(netparams.pop('activation'))
        if 'depth' and 'width' in netparams:
            str_params.append('{}x{}'.format(netparams.pop('depth'), netparams.pop('width')))

        if 'vendetta_activity_regularizer' in netparams:
            str_params.append('{:.4}'.format(netparams.pop('vendetta_activity_regularizer').l2))
        for key, value in netparams.items():
            try:
                str_params.append('{}:{}'.format(format_parameter_name(key), value.name))
            except AttributeError:
                if isinstance(value, dict):
                    value = get_param_str(value)
                    value = str(value).replace(' ', '')
                    str_params.append('{}:{}'.format(format_parameter_name(key), value))
                else:
                    str_params.append('{}:{}'.format(format_parameter_name(key), value))
    return str_params


def format_parameter_name(parameter_name):
    parts = parameter_name.split('_')
    return ''.join([part[0] for part in parts])


class Experiment:

    def set_session(self, seed=None, memory=None):
        if seed:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            rn.seed(seed)
            if memory:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory, allow_growth=True)
                session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                                              gpu_options=gpu_options)
            else:
                session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            tf.set_random_seed(seed)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)
        elif memory:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory, allow_growth=True)
            session_conf = tf.ConfigProto(gpu_options=gpu_options)
            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
            K.set_session(sess)

    def run(self, name=None, dataset='moons', lr=0.0001,
            batch_size=64, epochs=2000,
            optimizer='sgd', histogram_freq=0, write_grads=False, loss='crossentropy',
            epoch_start=0, epoch_freq=0, show_on_train=False, seed=None, show_input_layer=False,
            show_local_layers=False, show_decision_layers=True,
            batch_freq=0, check_weights=False, check_gradients=False, check_units=None, check_optimizer=False,
            write_graph=False, check_numerics=False, verbose=0, memory=None, write_images=False, embeddings_freq=None,
            embeddings_layer_names=None,
            extra=None, save_config=True, store_img=False, lr_schedule=None,
            min_lr=1e-5, max_lr=1e-2, min_mtm=0.85, max_mtm=0.95, step_size=1000, plot_matrix=False,
            plot_activations=False,
            plot_all_losses=False, log_dir='summaries', cached=False, plot_surface=False, single_plot=False,
            annealing_dropout_epochs=None, annealing_dropout_rate=0.5, print_weights=False, plot_epsilon=0.01,
            **kwargs):
        print(locals())
        network_parameters = kwargs
        # Stringify parameters of the experiment
        param_list = params2str(name, loss, optimizer, lr, batch_size, dataset, network_parameters, extra)
        # Address seed or memory fraction settings
        if seed or memory:
            self.set_session(seed, memory)

        # Load dataset
        (x_train, y_train), (x_test, y_test), classes = load_dataset(dataset)

        # Create model
        inputs = Input(x_train.shape[1:])
        outputs = self.get_network(network_parameters, inputs, classes, verbose=verbose)

        # Create loss and optimizer
        opt = keras.optimizers.get({'class_name': optimizer, 'config': {'lr': lr}})

        metrics = ['accuracy']
        if loss == 'hinge':
            y_train[y_train == 0] = -1
            y_test[y_test == 0] = -1
            if classes == 2:
                metrics = [binary_hinge_accuracy]

        elif loss == 'crossentropy':
            if classes > 2:
                loss = 'categorical_crossentropy'
                outputs = Activation('softmax')(outputs)
            else:
                loss = 'binary_crossentropy'
                outputs = Activation('sigmoid')(outputs)

        elif loss is None:
            def zero_loss(y_true, y_pred):
                return y_pred * 0.0

            loss = zero_loss
            outputs = Activation('softmax')(outputs)

        else:
            raise ValueError('Unknown loss {}'.format(loss))

        # Fit everything together
        model = Model(inputs=inputs, outputs=outputs, name=name)
        if verbose > 1:
            print(model.summary())

        # Get callbacks
        now = '{:%Y-%m-%d-%H:%M:%S}'.format(datetime.datetime.now())
        filename = slugify('-'.join(param_list + [now]))

        callbacks = get_callbacks(histogram_freq, write_grads, write_graph, write_images, epoch_freq,
                                  batch_freq, x_train, y_train, epoch_start, show_on_train, show_input_layer,
                                  show_local_layers,
                                  check_units, check_numerics, check_weights, check_optimizer, check_gradients,
                                  filename, store_img=store_img, embeddings_freq=embeddings_freq,
                                  embeddings_layer_names=embeddings_layer_names, plot_matrix=plot_matrix,
                                  log_dir=log_dir, show_decision_layers=show_decision_layers,
                                  plot_activations=plot_activations, plot_surface=plot_surface, single_plot=single_plot,
                                  annealing_dropout_epochs=annealing_dropout_epochs,
                                  annealing_dropout_rate=annealing_dropout_rate, print_weights=print_weights,
                                  plot_epsilon=plot_epsilon)

        if lr_schedule:
            def lr_metric(y_true, y_pred):
                return opt.lr

            metrics.append(lr_metric)
            if lr_schedule == 'sgdr':
                lr_scheduler = SGDRScheduler(min_lr=min_lr, max_lr=max_lr,
                                             steps_per_epoch=np.ceil(epochs / batch_size), )
            elif lr_schedule == 'clr_triangular':
                lr_scheduler = CyclicLR(base_lr=min_lr, max_lr=max_lr, mode='triangular')
            elif lr_schedule == 'clr_triangular2':
                lr_scheduler = CyclicLR(base_lr=min_lr, max_lr=max_lr, mode='triangular2')
            elif lr_schedule == 'clr_exp_range':
                lr_scheduler = CyclicLR(base_lr=min_lr, max_lr=max_lr, mode='exp_range')
            elif lr_schedule == 'clr':
                lr_scheduler = CLR(max_lr=max_lr, min_lr=min_lr, max_mtm=max_mtm, min_mtm=min_mtm, step_size=step_size)
            elif lr_schedule == 'lr_finder':
                lr_scheduler = LRFinder(step_size=batch_size, min_lr=min_lr, max_lr=max_lr)
            else:
                raise ValueError('Unknown lr_schedule {}'.format(lr_schedule))

            callbacks.append(lr_scheduler)

        if len(model.losses) > 1:
            def regularization_loss(y_true, y_pred):
                return K.sum(model.losses)

            metrics += [regularization_loss]

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=metrics)

        if save_config:
            config_path = os.path.join('config', filename + '.json')
            if not os.path.exists('config'):
                os.makedirs('config')
            config = model.to_json()
            with open(config_path, 'w') as f:
                f.write(config)

        file_template = slugify('-'.join(param_list)) + '*'
        exists = list(Path(log_dir).glob(file_template))

        if not cached or not exists:
            if verbose:
                print(f'Not found files for {file_template}, fitting')

            # Fit
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=False,
                      verbose=verbose,
                      callbacks=callbacks)
        else:
            if verbose:
                print(f'Found {len(exists)} files for {file_template}, skipping')
                if verbose > 1:
                    for f in exists:
                        print(f'\t{f}')

        if lr_schedule == 'lr_finder':
            lrs = pd.DataFrame(lr_scheduler.history)
            lrs.to_csv('lr_findings.csv')

        return model, callbacks

    def get_network(self, network_parameters, inputs, classes, verbose=0):
        raise NotImplementedError()
