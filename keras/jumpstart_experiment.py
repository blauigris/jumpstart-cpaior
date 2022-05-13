import numpy as np
from callbacks import HackyDropout
from joblib import Parallel, delayed
from keras import regularizers
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Activation
from sklearn.model_selection import ParameterGrid
import keras.backend as K

from experiment import Experiment
from jumpstart import JumpstartReLU


class SepConsExperiment(Experiment):
    def get_network(self, network_parameters, inputs, classes, verbose=0):
        if verbose > 1:
            print(network_parameters)
        outputs = inputs
        mixed = network_parameters.get('mixed', None)
        for i in range(network_parameters['depth']):
            # if i > 2:
            #     network_parameters['layer_parameters']['trainable'] = False
            if mixed is None:
                outputs = get_layer(outputs,
                                    network_parameters['activation'],
                                    network_parameters['width'],
                                    kernel_size=network_parameters['kernel_size'],
                                    layer_parameters=network_parameters['layer_parameters'])
            else:
                if mixed < i:
                    outputs = get_layer(outputs,
                                        network_parameters['activation'],
                                        network_parameters['width'],
                                        kernel_size=network_parameters['kernel_size'],
                                        layer_parameters=network_parameters['layer_parameters'])
                else:
                    outputs = get_layer(outputs,
                                        'lkrbf',
                                        network_parameters['width'],
                                        kernel_size=network_parameters['kernel_size'],
                                        layer_parameters=network_parameters['layer_parameters'])

        aggregation = network_parameters.get('aggregation', 'flatten')
        if aggregation == 'flatten':
            outputs = Flatten()(outputs)
        elif aggregation == 'avg_pool':
            outputs = GlobalAveragePooling2D()(outputs)
        outputs = Dense(classes, activation=None)(outputs) if classes > 2 else Dense(1, activation=None)(outputs)

        return outputs


def get_layer(inputs, activation, width, kernel_size=None, layer_parameters=None, verbose=0):
    if verbose > 2:
        print(layer_parameters)
    if kernel_size is None:
        kernel_size = inputs.shape[1:-1].as_list()

    if layer_parameters is None:
        layer_parameters = {}

    elif activation == 'relu':
        layer_parameters['activation'] = 'relu'
        print(layer_parameters)
        layer = Conv2D(width, kernel_size, **layer_parameters)(inputs)

    elif activation == 'relu_bn':
        layer_parameters['activation'] = None
        print(layer_parameters)
        layer = Conv2D(width, kernel_size, **layer_parameters)(inputs)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

    elif activation == 'separating_relu':
        lp = layer_parameters.copy()
        lp['activation'] = None
        separability_regularizer = lp.pop('separability_regularizer', None)
        axis = lp.pop('axis', None)
        balance = lp.pop('balance', None)
        print(lp)
        layer = Conv2D(width, kernel_size, **lp)(inputs)
        layer = JumpstartReLU(separability_regularizer=separability_regularizer, balance=balance, axis=axis)(layer)

    elif activation == 'separating_relu_annealing_dropout':
        lp = layer_parameters.copy()
        lp['activation'] = None
        separability_regularizer = lp.pop('separability_regularizer', None)
        axis = lp.pop('axis', None)
        rate = lp.pop('rate', 0)
        noise_shape = lp.pop('noise_shape', None)
        balance = lp.pop('balance', None)
        layer = inputs
        layer = HackyDropout(rate=rate, noise_shape=noise_shape)(layer)
        layer = Conv2D(width, kernel_size, **lp)(layer)
        layer = JumpstartReLU(separability_regularizer=separability_regularizer, balance=balance,
                              axis=axis)(layer)

    else:
        raise ValueError('Unknown layer {}'.format(activation))

    return layer


def run_50x4(lmda, axis, seed):
    sep_cons = SepConsExperiment()
    sep = regularizers.l2(lmda)
    sep.name = lmda
    sep_cons.run(name=f'10reps-seed-{seed}', dataset='moons', epochs=10000,
            optimizer='adam', lr=0.01, batch_size=85,
            # loss=None,
            depth=50, width=4, activation='separating_relu', kernel_size=None, seed=seed,
            layer_parameters={
                'separability_regularizer': sep,
                'use_bias': True,
                'axis': axis,
                # 'inverted': True,
                # 'balance': 0.2,
                # 'rate': 0

            },
            epoch_freq=100,
            # batch_freq=batch_freq,
            show_local_layers=False,
            show_decision_layers=False,
            store_img=False,
            show_contour=False,
            single_plot=False,
            verbose=0,
            plot_surface=False,
            )

def repetition_experiment():
    params = {'axis': [None, 0, -1, [0, -1]],
              'seed': np.arange(0, 10),
              'lmda': [1, 0.0001]
              }
    params = ParameterGrid(params)
    runs = []
    parallel = Parallel(n_jobs=4)
    for i in range(10):
        for p in params:
            axis = p['axis']
            seed = p['seed']
            lmda = p['lmda']
            runs.append(delayed(run_50x4)(lmda=lmda, axis=axis, seed=seed))

    output = parallel(runs)

def zero_test_experiment():
    lmda = 0.001
    sep = regularizers.l2(lmda)
    sep.name = lmda
    dropout_rate = 0.9
    name = f'zero'
    print(name)
    exp = SepConsExperiment()
    balance = 0.51
    exp.run(name=name, dataset='moons', epochs=100000,
            optimizer='adam', lr=0.01, batch_size=85,
            depth=50, width=4, activation='separating_relu_annealing_dropout',  seed=10,
            kernel_size=None,
            layer_parameters={
                'padding': 'same',

                'kernel_initializer': 'zeros',
                'bias_initializer': 'zeros',
                'balance': balance,

                'separability_regularizer': sep,
                'use_bias': True,
                'axis': [0,-1],
                'rate': K.variable(value=dropout_rate, dtype='float32', name='dropout_rate'),
            },
            annealing_dropout_epochs=1000,
            annealing_dropout_rate=dropout_rate,
            verbose=1,
            )


def glorot_experiment_test():
    exp = SepConsExperiment()
    lmda = 0.0001
    sep = regularizers.l2(lmda)
    sep.name = lmda
    exp.run(name='glorot', dataset='moons', epochs=100000,
            optimizer='adam', lr=0.01, batch_size=85,
            # loss=None,
            depth=50, width=4, activation='separating_relu', kernel_size=None, seed=10,
            layer_parameters={
                'separability_regularizer': sep,
                'use_bias': True,
                'axis': [0,-1],

            },
            epoch_freq=100,
            # Plotting internal representation options. plot_epsilon in the minimum change in accuracy to plot.
            show_local_layers=True,
            show_decision_layers=False,
            store_img=True,
            show_contour=False,
            single_plot=False,
            verbose=1,
            plot_surface=True,
            plot_epsilon=0.1
            )



if __name__ == '__main__':
    #repetition_experiment()
    zero_test_experiment()
    # glorot_experiment_test()
