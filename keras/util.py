import numpy as np
import seaborn as sns
from keras import backend as K, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

sns.set()


def binary_hinge_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)


def get_parameter_counts(model, line_length=None, positions=None):
    """Returns the number of parameters of keras model

    # Arguments
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
    """
    if model.__class__.__name__ == 'Sequential':
        sequential_like = True
    else:
        sequential_like = True
        for v in model.nodes_by_depth.values():
            if (len(v) > 1) or (len(v) == 1 and len(v[0].inbound_layers) > 1):
                # if the model has multiple nodes or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [.45, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #']
    else:
        line_length = line_length or 100
        positions = positions or [.33, .55, .67, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
        relevant_nodes = []
        for v in model.nodes_by_depth.values():
            relevant_nodes += v

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    return {'total': trainable_count + non_trainable_count, 'trainable': trainable_count,
            'non_trainable': non_trainable_count}


def dummy_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(4, (3, 3), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))
    return model
