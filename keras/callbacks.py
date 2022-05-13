import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.engine import Layer
from keras.layers import Dropout, interfaces
from keras.layers.convolutional import _Conv
from matplotlib import ticker, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
from slugify import slugify

warnings.filterwarnings("ignore")


class SurfaceCallback(Callback):
    def __init__(self, X, y, *, scale=1, title=None, show_ticks=True, resolution=100, show_colorbar=True, margin=0,
                 batch_freq=0, epoch_start=0, epoch_freq=1, show_on_train=True, show_input_layer=True,
                 show_local_layers=False,
                 show_decision_layers=False,
                 check_units=None, store_img=True, horizontal=False, output_dir='./plots', single_plot=True,
                 epsilon=0.05, print_weights=False):
        super().__init__()

        self.print_weights = print_weights
        self.single_plot = single_plot
        self.show_decision_layers = show_decision_layers
        self.output_dir = Path(output_dir)
        self.horizontal = horizontal
        self.store_img = store_img
        self.show_local_layers = show_local_layers
        self.check_units = check_units
        self.show_input_layer = show_input_layer
        self.epoch_start = epoch_start
        self.show_on_train = show_on_train
        self.epoch_freq = epoch_freq
        self.batch_freq = batch_freq
        self.X = X.squeeze()
        self.y = y.squeeze()
        self.margin = margin
        self.show_colorbar = show_colorbar
        self.scale = scale
        self.resolution = resolution
        self.show_ticks = show_ticks

        self.title = title

        self.img = None
        self._max = 0
        self._min = 0
        self._max_X = X.max()
        self._min_X = X.min()
        self.last_acc = 0
        self.last_val_acc = 0
        self.epsilon = epsilon

    def draw(self, epoch=None):
        layers = [l for l in self.model.layers if l.get_weights()]

        if self.show_local_layers:

            figs, axes = self.draw_local_layers(layers, epoch)
            fig, ax = plt.subplots()
            figs['output'] = fig
            axes['output'] = ax
            self._prepare_plot(ax, epoch)
            self.draw_contourf(fig, ax)
            self.draw_points(ax)
            if self.store_img:
                filename = slugify(self.title)
                output_dir = self.output_dir / f'local/{filename}/{epoch}'
                output_dir.mkdir(exist_ok=True, parents=True)
                for name, fig in figs.items():
                    fig.savefig(output_dir / f'{filename}-{name}.pdf')

                plt.close('all')

        # if self.show_decision_layers:
        #     if self.single_plot:
        #         nrows = np.ceil((len(layers) + 1) / 2).astype(np.int64)
        #         ncols = 2
        #         if self.horizontal:
        #             nrows, ncols = ncols, nrows
        #         figsize = np.array([ncols, nrows]) * 3
        #         fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        #         axes = axes.flatten()
        #
        #     else:
        #         figs = []
        #         axes = []
        #         for i in range(len(layers)):
        #             fig, ax = plt.subplots()
        #             figs.append(fig)
        #             axes.append(ax)
        #     self.draw_decision_layers(layers, axes, epoch)
        #     ax = axes[-1]
        #     self._prepare_plot(ax, epoch)
        #     self.draw_contourf(fig, ax)
        #     self.draw_points(ax)
        #     if self.store_img:
        #         filename = slugify(self.title)
        #         if self.single_plot:
        #             output_dir = self.output_dir / f'decision/{filename}/'
        #             output_dir.mkdir(parents=True, exist_ok=True)
        #             plt.savefig(output_dir / f'{filename}-{epoch}.pdf')
        #         else:
        #             output_dir = self.output_dir / f'local/{filename}/{epoch}'
        #             for i, fig in enumerate(figs):
        #                 fig.savefig(output_dir / f'{filename}-{i}.pdf')
        #
        # if not self.show_local_layers and not self.show_decision_layers:
        #     fig, ax = plt.subplots()
        #     self._prepare_plot(ax, epoch)
        #     self.draw_contourf(fig, ax)
        #     self.draw_points(ax)
        #     if self.store_img:
        #         filename = slugify(self.title)
        #         output_dir = self.output_dir / f'surface/{filename}/'
        #         output_dir.mkdir(parents=True, exist_ok=True)
        #         plt.savefig(output_dir / f'{filename}-{epoch}.pdf')
        # # if self.show_input_layer:
        # #     self.draw_input_layer(ax)

    def show(self, epoch=None):
        self.draw(epoch)
        if not self.store_img:
            plt.show()

    def draw_contourf(self, fig, ax):
        Z = self.decision_function(self.img)

        Z = Z.reshape(self.XX.shape)

        sns.set_palette('coolwarm')
        CS = ax.contourf(self.XX, self.YY, Z,
                         # vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)),
                         cmap=cm.coolwarm,
                         alpha=0.8)

        def fmt(x, pos):
            return '{:.2E}'.format(x)

        if self.show_colorbar:
            fig.colorbar(CS, ax=ax, alpha=0.7, format=ticker.FuncFormatter(fmt))

        ax.contour(self.XX, self.YY, Z, [-1, 0, 1], linewidth=(5, 10, 5),
                   colors=('blue', 'black', 'red'))

    def draw_points(self, ax):
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, vmin=-np.max(np.abs(self.y)), vmax=np.max(np.abs(self.y)),
                   cmap=cm.coolwarm, linewidths=1, edgecolors='black')

    def draw_input_layer(self, ax):
        weights = self.model.get_weights()[0]
        weights = np.reshape(weights, [weights.shape[0], weights.shape[-1]]).T
        ax.scatter(weights[:, 0], weights[:, 1], c='green',
                   linewidths=1, edgecolors='black')
        if self.check_units:
            ax.scatter(weights[self.check_units, 0], weights[self.check_units, 1], c='yellow', s=90,
                       linewidths=1, edgecolors='black')

    def _update_mesh(self):
        if self.show_input_layer:
            weights = self.model.layers[1].get_weights()[0]
            new_min = min(self._min_X, weights.min())
            new_max = max(self._max_X, weights.max())
            if self._min > new_min or self._max < new_max:
                num = self.resolution // self.X.ndim
                self._min = new_min
                self._max = new_max
                self._min = self._min * self.scale - self.margin
                self._max = self._max * self.scale + self.margin
                xxx = [np.linspace(self._min, self._max, num=num) for _ in range(self.X.ndim)]
                XXX = np.meshgrid(*xxx)

                self.img = np.c_[[xxx.ravel() for xxx in XXX]].T

                self.XX, self.YY = XXX
        elif self.img is None:
            num = self.resolution // self.X.ndim

            self._min = self._min_X * self.scale - self.margin
            self._max = self._max_X * self.scale + self.margin
            xxx = [np.linspace(self._min, self._max, num=num) for _ in range(self.X.ndim)]
            XXX = np.meshgrid(*xxx)

            self.img = np.c_[[xxx.ravel() for xxx in XXX]].T

            self.XX, self.YY = XXX

    def _prepare_plot(self, ax, epoch=None):
        self._update_mesh()
        if not self.show_local_layers:
            if epoch:
                ax.set_title('Epoch {} '.format(epoch) + self.title)
            else:
                ax.set_title(self.title)
        sns.set_context("paper")
        sns.set_style("whitegrid")
        # sns.set_style("white")
        sns.despine()

    def decision_function(self, X):
        X = X[:, :, np.newaxis, np.newaxis]
        X = X.astype('float32')
        pred = self.model.predict(X)
        return np.squeeze(pred)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.epoch_start and self.epoch_freq and epoch % self.epoch_freq == 0:
            self.show(epoch)

        if np.abs(self.last_acc - logs['acc']) > self.epsilon or np.abs(self.last_val_acc - logs['val_acc']) > self.epsilon:
            if self.last_val_acc != logs['val_acc']:
                self.last_val_acc = logs['val_acc']
            if self.last_acc != logs['acc']:
                self.last_acc = logs['acc']
            self.show(epoch)


    def on_train_begin(self, logs=None):
        if self.show_on_train:
            self.show(epoch='START')

    def on_train_end(self, logs=None):
        if self.show_on_train:
            self.show(epoch='END')

    def draw_local_layers(self, layers, single_plot=False, epoch=None):
        # self._2d_fig.suptitle('Epoch {} '.format(epoch) + self.title)
        X = self.X[:, :, np.newaxis, np.newaxis]
        X = X.astype('float32')

        sns.set_context("paper")
        # sns.set_style("whitegrid")
        sns.set_style("white")
        sns.despine()

        figs = {}
        axes = {}
        for i, layer in enumerate(layers):
            weights = layer.get_weights()

            if weights:
                # ax.set_title('{}'.format(layer.name.split()))
                input_ = layer.input.eval(session=K.get_session(), feed_dict={self.model.inputs[0]: X})
                input_ = input_.squeeze()
                kernel = weights[0].squeeze().T
                if kernel.ndim == 1:
                    kernel = kernel[np.newaxis, :]

                try:
                    bias = weights[1] if layer.use_bias else None
                    if self.print_weights:
                        print(kernel)
                        print(bias)
                except AttributeError:
                    bias = None

                # print(layer.name, input_.shape, kernel.shape)
                for start in range(0, input_.shape[1], 2):
                    fig, ax = self._draw_layer_2d(input_[:, [start, start + 1]], kernel[:, [start, start + 1]], bias)
                    name = f'{layer.name}-{start}'
                    figs[name] = fig
                    axes[name] = ax

        return figs, axes

                # input_ = input_.reshape((input_.shape[0] * input_.shape[-1], 2))


    def _draw_layer_2d(self, input_, kernel, bias=None):
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        fig, ax = plt.subplots()
        if input_.ndim == 1:
            ax.scatter(input_, np.zeros_like(input_), c=self.y.squeeze(), cmap=cmap, alpha=0.8)
        elif input_.ndim == 2:
            ax.scatter(input_[:, 0], input_[:, 1], c=self.y.squeeze(), cmap=cmap, alpha=0.8)
        else:
            raise RuntimeError('Unexpected input dimensionality')

        ax.autoscale(enable=False, axis='both', tight=None)
        bottom_left = ax.transData.inverted().transform(ax.transAxes.transform([0, 0]))
        bottom_right = ax.transData.inverted().transform(ax.transAxes.transform([1, 0]))
        top_left = ax.transData.inverted().transform(ax.transAxes.transform([0, 1]))
        top_right = ax.transData.inverted().transform(ax.transAxes.transform([1, 1]))
        ax_vertices = [bottom_left, top_left, top_right, bottom_right]
        x_plane = [bottom_left[0], bottom_right[0]]
        if bias is not None:
            y_planes = (-bias[:, np.newaxis] - kernel[:, 0][:, np.newaxis] @ [x_plane]) / kernel[:, 1][:,
                                                                                          np.newaxis]
        else:
            y_planes = (- kernel[:, 0][:, np.newaxis] @ [x_plane]) / kernel[:, 1][:, np.newaxis]
            bias = np.zeros(kernel.shape[0])

        # current_palette = list(sns.color_palette('deep'))
        c = 'b'

        for y_plane, w, b in zip(y_planes, kernel, bias):
            ax.plot(x_plane, y_plane, clip_on=True, c=c)
            plane_points = np.vstack((x_plane, y_plane)).T
            other_vertices = []
            for vertex in ax_vertices:
                if w @ vertex + b <= 0:
                    other_vertices.append(vertex)
            if len(other_vertices) > 0:
                zero_vertices = np.vstack((plane_points, np.vstack(other_vertices)))

                # Sort vertices
                # compute centroid
                cent = zero_vertices.mean(axis=0)
                order = zero_vertices - cent
                order = np.arctan2(order[:, 0], order[:, 1])
                order = np.argsort(order)
                zero_patch = Polygon(zero_vertices[order], color='gray', alpha=0.25, closed=True, clip_on=True)
                ax.add_patch(zero_patch)

        return fig, ax

    def draw_decision_layers(self, layers, axes, epoch=None):

        color_palettes = [
            sns.diverging_palette(240, 10, n=9, as_cmap=True),
            sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True),
            sns.diverging_palette(10, 220, sep=80, n=9, as_cmap=True),
            sns.diverging_palette(145, 280, s=85, l=25, n=9, as_cmap=True)
        ]
        for i, (layer, ax) in enumerate(zip(layers, axes)):
            self._update_mesh()
            ax.set_title('{}'.format(layer.name))

            Z = layer.output.eval(session=K.get_session(),
                                  feed_dict={self.model.input: self.img[..., np.newaxis, np.newaxis]}).squeeze()

            if Z.ndim > 1:
                for i in range(Z.shape[1]):
                    Z_ax = Z[:, i].reshape(self.XX.shape)

                    CS = ax.contourf(self.XX, self.YY, Z_ax,
                                     # vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)),
                                     cmap=color_palettes[i],
                                     alpha=0.2)

                    ax.contour(self.XX, self.YY, Z_ax, [-1, 0, 1], linewidth=(5, 10, 5),
                               colors=('blue', 'black', 'red'))
            else:
                Z_ax = Z.reshape(self.XX.shape)

                CS = ax.contourf(self.XX, self.YY, Z_ax,
                                 # vmin=-np.max(np.abs(Z)), vmax=np.max(np.abs(Z)),
                                 cmap=color_palettes[0],
                                 alpha=0.75)

                ax.contour(self.XX, self.YY, Z_ax, [-1, 0, 1], linewidth=(5, 10, 5),
                           colors=('blue', 'black', 'red'))

            self.draw_points(ax)


def circles(ax, x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles.
    Similar to plt.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax.add_collection(collection)

    return collection


class NanAlert(Callback):

    def __init__(self, X, y, *, check_weights=False, check_gradients=False, check_units=None, check_optimizer=False,
                 check_numerics=False):
        super().__init__()
        self.check_optimizer = check_optimizer
        self.check_units = check_units
        self.check_gradients = check_gradients
        self.check_weights = check_weights
        self.y = y
        self.X = X
        self._gradient_tensors = None
        self.check_numerics = check_numerics

    def on_epoch_begin(self, epoch, logs=None):
        if self.check_weights:
            self._check_weights(epoch)
        if self.check_gradients:
            self._check_gradients(epoch)
        if self.check_units:
            self._check_units(epoch)
        if self.check_optimizer:
            self._check_optimizer(epoch)

    def on_batch_end(self, batch, logs=None):
        if self.check_numerics:
            K.get_session().run(self.check_numerics_op, feed_dict={self.model.inputs[0]: self.X,
                                                                   self.model.targets[0]: self.y,
                                                                   self.model.sample_weights[0]: np.ones(
                                                                       self.y.shape[0]),
                                                                   })

    def set_model(self, model):
        super().set_model(model)
        if self.check_numerics:
            self.check_numerics_op = tf.add_check_numerics_ops()

    def _check_units(self, epoch, logs=None):
        print('Units {} at batch {}'.format(self.check_units, epoch), file=sys.stderr)
        weigths = self.model.layers[1].kernel.eval(session=K.get_session())
        if self.model.layers[1].use_bias:
            bias = self.model.layers[1].bias.eval(session=K.get_session())
        if self.check_units == 'all':
            fuck_you_weights = weigths.squeeze()
            if self.model.layers[1].use_bias:
                fuck_you_bias = bias.squeeze()
        else:
            fuck_you_weights = weigths.squeeze()[:, self.check_units]
            if self.model.layers[1].use_bias:
                fuck_you_bias = bias.squeeze()[self.check_units]
        print(fuck_you_weights, file=sys.stderr)
        if self.model.layers[1].use_bias:
            print(fuck_you_bias, file=sys.stderr)

    def _check_weights(self, batch):
        output = self.model.predict(self.X)
        is_nan = np.isnan(output).any()
        is_inf = np.isinf(output).any()
        if is_nan:
            print('NAN DETECTED IN BATCH {}'.format(batch), file=sys.stderr)
        if is_inf:
            print('INF DETECTED IN BATCH {}'.format(batch), file=sys.stderr)

        for weight in self.model.weights:
            value = weight.eval(session=K.get_session())
            if np.isnan(value).any() or np.isinf(value).any():
                fuck_you = np.nonzero(np.isnan(value))
                print('ITS YOUR FAULT {} - {}'.format(weight.name, fuck_you), file=sys.stderr)

    def _check_gradients(self, epoch):
        # Lazy init
        if self._gradient_tensors is None:
            self._gradient_tensors = self.model.optimizer.get_updates(self.model.total_loss,
                                                                      self.model.trainable_weights)

        # Check next batch for nans/infs
        gradients = K.get_session().run(self._gradient_tensors,
                                        feed_dict={self.model.inputs[0]: self.X,
                                                   self.model.targets[0]: self.y,
                                                   self.model.sample_weights[0]: np.ones(self.y.shape[0]),
                                                   })

        vars_and_gradients = zip(self.model.trainable_weights, gradients)
        for var, gradient in vars_and_gradients:
            is_nan = np.isnan(gradient).any()
            is_inf = np.isinf(gradient).any()
            if is_nan or is_inf:
                fail = 'nan' if is_nan else 'inf'
                print('ALARM, ALARM! Possible nan in gradients at epoch {} due {} in {}'.format(epoch, fail, var.name),
                      file=sys.stderr)
                culprit = np.nonzero(np.isnan(gradient))
                print(culprit, file=sys.stderr)

    def _check_optimizer(self, epoch):
        # Check next batch for nans/infs
        optimizer_values = K.get_session().run(self.model.weights)

        vars_and_values = zip(self.model.optimizer.weights, optimizer_values)
        for var, values in vars_and_values:
            is_nan = np.isnan(values).any()
            is_inf = np.isinf(values).any()
            if is_nan or is_inf:
                fail = 'nan' if is_nan else 'inf'
                print('ALARM, ALARM! Possible nan in optimizer variable {} at epoch {} due {}'.format(var.name, epoch,
                                                                                                      fail),
                      file=sys.stderr)
                culprit = np.nonzero(np.isnan(values))
                print(culprit, file=sys.stderr)


class MatrixPlotNetwork(Callback):

    def __init__(self, batch_freq=None, epoch_freq=None, title='matrix', store_img=False, output_dir='./plots/'):
        super().__init__()
        self.output_dir = output_dir
        self.store_img = store_img
        self.title = title
        self.batch_freq = batch_freq
        self.epoch_freq = epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_freq and epoch % self.epoch_freq == 0:
            self.update_epoch(epoch)

    def on_batch_end(self, batch, logs=None):
        if self.batch_freq and batch % self.batch_freq == 0:
            self.update_batch(batch)

    def update_epoch(self, epoch):
        weights = self._get_weights()
        ax = sns.heatmap(weights)
        ax.set_title(f'Epoch {epoch}')
        if self.store_img:
            filename = slugify(self.title)
            output_dir = self.output_dir / f'matrix/{filename}/'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f'{filename}-{epoch}.pdf')
        else:
            plt.show()

    def update_batch(self, batch):
        weights = self._get_weights()
        ax = sns.heatmap(weights)
        ax.set_title(f'Batch {batch}')
        plt.show()
        if self.store_img:
            filename = slugify(self.title)
            output_dir = self.output_dir / f'matrix/{filename}/'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f'{filename}-{batch}.pdf')
        else:
            plt.show()

    def _get_weights(self):
        layers = []
        for layer in self.model.layers:
            if isinstance(layer, _Conv):

                if layer.use_bias:
                    w, b = layer.get_weights()
                    w = w.reshape(w.shape[-1], -1)
                    weights = np.vstack((w.T, b)).T
                    unit_names = [f'{layer.name}-{i}' for i in range(weights.shape[0])]
                    parameter_names = [f'w{i}' for i in range(weights.shape[-1])] + ['b']


                else:
                    w = layer.get_weights()[0]
                    weights = w.reshape(w.shape[-1], -1)
                    unit_names = [f'{layer.name}-{i}' for i in range(weights.shape[0])]
                    parameter_names = [f'w{i}' for i in range(weights.shape[-1])]
                print(w.shape)

                weights = pd.DataFrame(weights, index=unit_names, columns=parameter_names)
                layers.append(weights)
        return pd.concat(layers)


class ActivationPlotNetwork(Callback):

    def __init__(self, X, batch_freq=None, epoch_freq=None, title='matrix', store_img=False, output_dir='./plots/'):
        super().__init__()
        self.cbar = True
        self.X = X
        self.output_dir = Path(output_dir)
        self.store_img = store_img
        self.title = title
        self.batch_freq = batch_freq
        self.epoch_freq = epoch_freq

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_freq and epoch % self.epoch_freq == 0:
            self.update_epoch(epoch)

    def on_batch_end(self, batch, logs=None):
        if self.batch_freq and batch % self.batch_freq == 0:
            self.update_batch(batch)

    def update_epoch(self, epoch):
        self.draw(epoch)

    def update_batch(self, batch):
        self.draw(batch)

    def draw(self, idx):
        labels, activations = self._get_activations()
        # v = np.max(np.abs(activations))
        # fig, axes = plt.subplots(1, len(activations))
        # axes = axes.flatten()
        # for ax, activations in zip(axes, activations):
        #     sns.heatmap(activations, vmin=-15, vmax=15, ax=ax, cmap='coolwarm')
        fig, ax = plt.subplots(1, figsize=(25, 15))
        activations = np.hstack(activations)
        sns.heatmap(activations, vmin=-15, vmax=15, cmap='coolwarm', cbar=self.cbar, ax=ax)
        ax.set_xticks(np.arange(2, len(activations) * 4, 4))
        ax.set_xticklabels(labels)
        plt.tight_layout()
        self.cbar = False
        if self.store_img:
            filename = slugify(self.title)
            output_dir = self.output_dir / f'activation/{filename}/'
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f'{filename}-{idx}.pdf')

            # fig.savefig(output_dir / f'{filename}-{idx}.pdf')
        else:
            plt.show()

    def _get_activations(self):
        activations = []
        labels = []
        for layer in self.model.layers:
            try:
                if layer.activation.__name__.endswith('relu'):
                    activation = layer.output.eval(session=K.get_session(),
                                                   feed_dict={self.model.input: self.X}).squeeze()

                    if activation.ndim > 1 and activation.shape[1] == 4:
                        activations.append(activation)
                        labels.append(layer.name)
                        # print(activation.shape)
            except AttributeError as ex:
                pass

        return labels, activations


class AnnealingDropoutCallback(Callback):
    def __init__(self, epochs, rate=1.0):
        super().__init__()
        self.rate = rate
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs={}):
        v = max(0, self.rate - self.rate * epoch / self.epochs)
        print("epoch", epoch, "update", v)
        for layer in self.model.layers:

            if type(layer) == HackyDropout:
                K.set_value(layer.rate, v)


class HackyDropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """
    #@interfaces.legacy_dropout_support
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(HackyDropout, self).__init__(**kwargs)
        # self.rate = min(1., max(0., rate))
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < K.eval(self.rate) < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

    def get_config(self):
        config = {'rate': K.eval(self.rate),
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(HackyDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
