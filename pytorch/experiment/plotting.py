from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.core.dtypes.common import is_timedelta64_dtype

from experiment.experiment import simple_encode_params
from jumpstart.util import weights_to_df


def plot_grid(ax, grid, title='', vmin=None, vmax=None, fmt=None, annot=True, cbar_ax=None, xticklabels='auto',
              yticklabels='auto'):
    if isinstance(vmax, pd.Timedelta):
        vmax = vmax.seconds / 60.0  # Minutes

    if isinstance(vmin, pd.Timedelta):
        vmin = vmin.seconds / 60.0  # Minutes

    if is_timedelta64_dtype(grid.dtypes.iloc[0]):
        grid = grid.astype('timedelta64[m]')

    bad = pd.isna(grid)
    grid = grid.fillna(0)
    grid = grid.apply(lambda x: pd.to_numeric(x, downcast='integer'))

    # try:
    #     grid = grid.astype('Int64')
    # except TypeError:
    #     pass

    if grid.dtypes.apply(pd.api.types.is_integer_dtype).all():
        fmt = 'd'
    else:
        fmt = ".2g" if not fmt else fmt

    sns.heatmap(grid, mask=bad, annot=annot, square=True, ax=ax, vmin=vmin, vmax=vmax, fmt=fmt, cbar_ax=cbar_ax,
                xticklabels=xticklabels, yticklabels=yticklabels)

    ax.set_title(title)
    ax.set_ylabel('Width')
    ax.set_xlabel('Depth')
    ax.invert_yaxis()


def plot_grids(grids, rows, metrics=None, normalize=False,
               figsize=None, scale=4, show_title=False):
    if metrics is None:
        metrics = grids.columns

    for metric in metrics:
        tag_grids = grids[metric]
        if normalize:
            vmax, vmin = max([grid.max().max() for grid in grids[metric]]), 0
        else:
            vmax, vmin = None, None

        title = metric if show_title else None
        plot_grids_tag(tag_grids, rows, vmin=vmin, vmax=vmax, figsize=figsize, scale=scale, title=title)


def plot_grids_tag(grids, rows, vmin=None, vmax=None, figsize=None, scale=4, title=None):
    grids_groupby = grids.groupby(rows)

    n_rows = len(grids_groupby)
    n_cols = max(len(v) for k, v in grids_groupby.groups.items())
    figsize = figsize if figsize else (scale * n_rows, scale * n_cols * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title)

    for (_, row_grids), row_axes in zip(grids_groupby, axes):
        if n_cols == 1:
            row_axes = [row_axes]
        for (params, grid), ax in zip(row_grids.iteritems(), row_axes):
            title = '-'.join([f'{k}:{v}' for k, v in zip(grids.index.names, params)])
            try:
                plot_grid(ax, grid.T, title=title, vmin=vmin, vmax=vmax)
            except ValueError as ex:
                print(f'Failed {title} due to {ex}')

    plt.tight_layout()
    return fig, axes


def plot_layer_sets_table(ax, jumpstart_metrics):
    table = jumpstart_metrics.layer_sets_table
    sns.heatmap(table.astype(int), ax=ax,
                vmin=-1, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["#db5e56", "#5ea28d", "#dbc256", ],
                                                                         N=3),
                mask=table.isnull(),
                square=True,
                fmt='d'
                )

    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.set_ticklabels(['Zero', 'Intersection', 'Affine'])


def plot_dead_units(ax, jumpstart_metrics, output_file=None, square=True, xticklabels=True, yticklabels=True,
                    cbar_ax=None):
    all_unit_table = jumpstart_metrics.all_unit_table
    if jumpstart_metrics.all_unit_table.isna().to_numpy().any():
        all_unit_table = all_unit_table.astype(float)

    sns.heatmap(all_unit_table, ax=ax,
                vmin=-1, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["#db5e56", "#5ea28d", "#dbc256", ],
                                                                         N=3),
                mask=jumpstart_metrics.all_unit_table.isnull(),
                square=square,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar_ax=cbar_ax,
                )

    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.set_ticklabels(['Dead', 'Alive', 'Affine'])
    plt.tight_layout()

    if output_file:
        filename = Path(output_file)
        filename.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(str(filename))
        plt.close(ax.figure)


def plot_dead_points(ax, jumpstart_metrics, output_file=None, square=True, xticklabels=True, yticklabels=True,
                     cbar=True, cbar_ax=None):
    sns.heatmap(jumpstart_metrics.layer_sets_table, ax=ax,
                vmin=-1, vmax=1,
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["#db5e56", "#5ea28d", "#dbc256", ],
                                                                         N=3),
                mask=jumpstart_metrics.layer_sets_table.isnull(),
                square=square,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=cbar,
                cbar_ax=cbar_ax
                )
    if cbar:
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
        colorbar.set_ticklabels(['Dead', 'Alive', 'Affine'])
    plt.tight_layout()

    # ax.set_xlim([0, len(jumpstart_metrics.layer_sets)])
    # ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    # ax.axis('off')
    if output_file:
        filename = Path(output_file)
        filename.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(str(filename))
        plt.close(ax.figure)


def plot_dead_units_points(fig, jumpstart_metrics, output_file=None, square=True, yticklabels=True, xticklabels=True):
    gs = GridSpec(2, 2, width_ratios=[20, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    plot_dead_points(ax1, jumpstart_metrics, cbar=False, square=False, xticklabels=xticklabels,
                     yticklabels=yticklabels)
    plot_dead_units(ax2, jumpstart_metrics, cbar_ax=ax3, square=False, xticklabels=False, yticklabels=yticklabels)


def plot_decision_surface(ax, X, y, model=None, h=.2, show_colorbar=True, show_ticks=False, point_labels=None,
                          levels=10):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = 'coolwarm'
    cm_bright = ListedColormap(['#0000FF', '#FF0000', ])

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if not show_ticks:
        ax.set_xticks(())
        ax.set_yticks(())
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if model:
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = model(torch.tensor(X_mesh, dtype=torch.float32))

        # Put the result into a color plot
        Z = Z.reshape(xx.shape).detach().numpy()
        Z_max = np.abs(Z).max()
        cs_contourf = ax.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=.8, vmin=-Z_max, vmax=Z_max, )
        cs_contour = ax.contour(xx, yy, Z, [-1, 0, 1], linewidths=[1, 2, 1], alpha=0.5,
                                colors=('#0000FF', 'gray', '#FF0000'))

        if show_colorbar:
            # Make a colorbar for the ContourSet returned by the contourf call.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = ax.figure.colorbar(cs_contourf, cax=cax)
            # Add the contour line levels to the colorbar
            cbar.add_lines(cs_contour)

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k')

    if point_labels is not None:
        point_labels = range(X.shape[0]) if point_labels is True else point_labels
        for i, point_label in enumerate(point_labels):
            ax.annotate(f'{point_label}', (X[i, 0] + 0.05, X[i, 1]), fontsize='xx-large')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    if not show_ticks:
        ax.set_xticks(())
        ax.set_yticks(())
    ax.set_aspect('equal', 'box')


def plot_layer_sets(ax, jumpstart_metrics, epoch=None, orientation='horizontal'):
    # Add ranks to see zombie points
    layer_indexes = {}
    layer_colors = {}
    for layer_name, layer_sets in jumpstart_metrics.layer_sets.items():
        indexes = np.concatenate(layer_sets)
        layer_indexes[layer_name] = indexes
        colors = np.zeros(len(indexes))
        colors[:len(layer_sets[0])] = -1
        colors[-len(layer_sets[2]):] = 1
        layer_colors[layer_name] = colors

    layer_indexes = pd.DataFrame.from_dict(layer_indexes, orient='index').T
    layer_colors = pd.DataFrame.from_dict(layer_colors, orient='index').T

    if orientation == 'vertical':
        layer_colors, layer_indexes = layer_indexes.T, layer_colors.T
    sns.heatmap(layer_colors, ax=ax,
                square=True,
                annot=layer_indexes,
                # cbar=False,
                center=0,
                vmin=-1, vmax=1,
                fmt='d'
                )


def plot_activations(ax, jumpstart_metrics):
    sns.heatmap(jumpstart_metrics.output_table,
                # square=True,
                robust=True,
                ax=ax)


def plot_binary_activations(ax, jumpstart_metrics):
    sns.heatmap(jumpstart_metrics.output_table > 0,
                # square=True,
                fmt='d',
                ax=ax)


def plot_training_summary(jumpstart_metrics, permutation_identifiers=None, n_colors=None, epoch=None, logs=None,
                          figsize=(25, 15), mode='positive', correction=None, output_dir=None):
    permutation_identifiers = permutation_identifiers if permutation_identifiers else {}
    irreducible_model, original_permutations = jumpstart_metrics.compress(mode=mode,
                                                                           adjustment=correction,
                                                                           permutation_identifiers=permutation_identifiers)

    X, y = next(iter(jumpstart_metrics.dataloader))
    y = y.numpy().flatten()

    # Plot
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec[0, :])
    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[1, 1])
    ax4 = fig.add_subplot(spec[1, 2])
    ax5 = fig.add_subplot(spec[1, 3])

    plot_network_permutation_diagram(ax1, original_permutations, irreducible_model.graph, n_colors=n_colors)
    plot_equivalence_descriptors(ax2, y, permutation_identifiers, mode)
    plot_irreducible_graph(ax3, irreducible_model.graph, n_colors=len(permutation_identifiers))
    plot_decision_surface(ax4, X, y, jumpstart_metrics.model)
    plot_decision_surface(ax5, X, y, model=irreducible_model)

    if output_dir:
        filename = Path(f'./{output_dir}/jumpstart-epoch-{epoch}.pdf')
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(filename))
        plt.close(fig)
    else:
        plt.show()


def plot_partitions(ax, X, y, partitions, solution=None, h=.02, show_colorbar=True, point_labels=None):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = 'coolwarm'
    cm_bright = ListedColormap(['#0000FF', '#FF0000', ])

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    # Plot the planes stored in target_partition
    for partition in partitions:
        x_plane = np.arange(x_min, x_max, h)
        y_plane = partition(x_plane)
        ax.plot(x_plane, y_plane)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if solution:
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = solution(torch.tensor(X_mesh, dtype=torch.float32))

        # Put the result into a color plot
        Z = Z.reshape(xx.shape).detach().numpy()
        Z_max = np.abs(Z).max()
        cs_contourf = ax.contourf(xx, yy, Z, cmap=cm, alpha=.8, vmin=-Z_max, vmax=Z_max)
        cs_contour = ax.contour(xx, yy, Z, [0], linewidths=1, alpha=0.5,
                                colors=('gray',))

        if show_colorbar:
            # Make a colorbar for the ContourSet returned by the contourf call.
            cbar = ax.figure.colorbar(cs_contourf, ax=ax)
            # Add the contour line levels to the colorbar
            cbar.add_lines(cs_contour)

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k')

    point_labels = range(X.shape[0]) if point_labels is None else point_labels
    for i, point_label in enumerate(point_labels):
        ax.annotate(f'{point_label}', (X[i, 0] + 0.05, X[i, 1]), fontsize='xx-large')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect('equal', 'box')


figure_template = r"""
\begin{{figure}}[ht]
  \begin{{center}}
    {content}
  \end{{center}}
  \caption{{{caption}}}
  \label{{fig:{label}}} 
\end{{figure}}
"""

subfigure_template = r"""\begin{{subfigure}}[t]{{{width}\textwidth}}
        \includegraphics[width=\textwidth]{{{filepath}}}
        \caption{{{caption}}}
        \label{{fig:{label}}}
    \end{{subfigure}}
    ~"""


def get_width_depth_intervals(grids):
    intervals = []
    for metric_name, metric_grid in grids.items():
        intervals.append(
            metric_grid.apply(lambda x: pd.Series((min(x.index), max(x.index), min(x.columns), max(x.columns)),
                                                  index=['min_width', 'max_width', 'min_depth', 'max_depth'])))

    return pd.concat(intervals)


def generate_subfigure_caption(params, metric):
    metric = metric.replace('_', ' ').replace('acc', 'accuracy').replace('init', 'initialization').capitalize()
    return f'{metric}'


def generate_figure_caption(params, min_width, max_width, min_depth, max_depth, dataset):
    figure_caption = f'Depth vs width accuracy heatmap a for a grid of rectangular networks with width ' \
                     f'from ${min_width}$ to ${max_width}$ and depth from ${min_depth}$ to ${max_depth}$, trained using ' \
                     f'{params["optimizer"]} with a learning rate of {params["lr"]} for {params["epochs"]} epochs in the {dataset} dataset.'

    if params["lambda"] > 0:
        figure_caption += f', and \ResurrectionConsUP with a $\lambda$ of {params["lambda"]} ' \
                          f'and using {params["aggr"]} as aggregation.'
    else:
        figure_caption += ', with no \ResurrectionConsUP as baseline.'

    return figure_caption


def generate_subfigure_label(config, metric, dataset):
    if config['lambda'] > 0:
        label = f'{dataset}_full_grid_' + '_'.join([f'{k}_{v}' for k, v in config.items()]) + f'-{metric}'
    else:
        if 'lr' in config:
            label = f'{dataset}_full_grid_baseline_lr_{config["lr"]}-{metric}'
        else:
            label = f'{dataset}_full_grid_baseline-{metric}'

    return label.replace('.', '').replace('_00_', '_0_')


def generate_figure_label(config, dataset):
    if config['lambda'] > 0:
        label = f'{dataset}_full_grid_' + '_'.join([f'{k}_{v}' for k, v in config.items()])
    else:
        if 'lr' in config:
            label = f'{dataset}_full_grid_baseline_lr_{config["lr"]}'
        else:
            label = f'{dataset}_full_grid_baseline'

    return label.replace('.', '').replace('_00_', '_0_')


def export_grid_plots(grids, output_dir=None, figsize=None, vmin=None, vmax=None, fmt=None,
                      width=None, latex_path_prefix='', annot=True, cbar_figsize=None,
                      xticklabels='auto', yticklabels='auto', format='pdf'):
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = grids.reset_index()[grids.index.names]
    hparam_counts = hparams.apply(lambda x: x.unique().shape[0])
    dataset = hparams['dataset'].iloc[0]
    common_hparams = hparams.columns[hparam_counts == 1].to_list()
    different_hparams = hparams.columns[hparam_counts > 1].to_list()
    common_config = dict(zip(common_hparams, hparams.loc[0, common_hparams]))
    encoded_common_params = simple_encode_params(common_config)
    if len(encoded_common_params) > 80:
        encoded_common_params = encode_params_str(common_config)
    common_dir = output_dir / encoded_common_params
    common_dir.mkdir(parents=True, exist_ok=True)

    latex_dir = common_dir / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)
    hparam_grids = grids.reset_index().drop(columns=common_hparams).set_index(different_hparams)
    intervals = get_width_depth_intervals(grids)
    intervals = intervals.drop_duplicates().iloc[0]
    for params, metric_grids in hparam_grids.iterrows():
        latex_subfigures = []
        config = dict(zip(hparam_grids.index.names, params)) if len(hparam_grids.index.names) > 1 else {
            hparam_grids.index.names[0]: params}
        config_str = simple_encode_params(config)
        print(config)
        for metric, metric_grid in metric_grids.iteritems():
            fig, ax = plt.subplots(1, figsize=figsize)
            cbar_fig, cbar_ax = plt.subplots(1, figsize=cbar_figsize)
            plot_grid(ax, metric_grid, vmin=vmin, vmax=vmax, fmt=fmt, annot=annot, cbar_ax=cbar_ax,
                      xticklabels=xticklabels, yticklabels=yticklabels)
            fig.tight_layout()
            filepath = common_dir / f'{config_str}-{metric}.{format}'
            fig.savefig(filepath)
            cbar_fig.tight_layout()
            cbar_filepath = common_dir / f'cbar_{config_str}-{metric}.{format}'
            cbar_fig.savefig(cbar_filepath)
            latex_filepath = f'{latex_path_prefix}/{config_str}-{metric}.{format}'
            subfigure_caption = generate_subfigure_caption(config, metric=metric)
            subfigure_label = generate_subfigure_label(config=config, metric=metric, dataset=dataset)
            subfigure = subfigure_template.format(filepath=latex_filepath,
                                                  width=width if width is not None else 1 / hparam_grids.shape[
                                                      1] - 0.01,
                                                  caption=subfigure_caption,
                                                  label=subfigure_label)
            latex_subfigures.append(subfigure)
            plt.close(fig)
            plt.close(cbar_fig)
        latex_subfigures_text = '\n'.join(latex_subfigures)
        full_config = {**common_config, **config}
        figure_caption = generate_figure_caption(params=full_config,
                                                 min_depth=intervals['min_depth'],
                                                 max_depth=intervals['max_depth'],
                                                 min_width=intervals['min_width'],
                                                 max_width=intervals['max_width'],
                                                 dataset=dataset
                                                 )
        figure_label = generate_figure_label(config, dataset)
        latex_figure_text = figure_template.format(content=latex_subfigures_text, caption=figure_caption,
                                                   label=figure_label)

        with open(latex_dir / f'{config_str}.tex', 'w') as f:
            f.write(latex_figure_text)


def plot_weights(ax, model, epoch=None):
    table = weights_to_df(model)
    sns.heatmap(table,
                square=True,
                robust=True,
                ax=ax,
                )

    if epoch:
        ax.set_title(f'Epoch {epoch}')

    return model


def plot_param_arch(ax, data, x_key, y_key, groupby, ylabel=None, label_fmt=None, show_arch=True, colors=None,
                    markers=None, architectures=None, legend_ax=None):
    sns.set_style('whitegrid')

    ax.set_xscale('log')
    groups = [i for i in data.groupby(groupby)]
    for index, group in reversed(groups):
        group = group.sort_values('trainable_params')
        label = label_fmt(index) if label_fmt else dict(zip(groupby, index))
        color = colors[(index[1], index[2])] if colors else None
        marker = markers[(index[1], index[0])] if (index[1], index[0]) in markers else None
        ax.scatter(group[x_key], group[y_key], alpha=0.5,
                   label=label,
                   color=color,
                   marker=marker
                   )

    if show_arch:
        dxw_labels = data[[x_key, 'depth', 'width']].drop_duplicates().sort_values(x_key)
        dxw_labels['labels'] = dxw_labels[['depth', 'width']].astype(str).agg('x'.join, axis=1)
        if architectures:
            not_selected = ~dxw_labels['labels'].isin(architectures)
            dxw_labels.loc[not_selected, 'labels'] = ''
        linestyles = [':', '--', '-']
        for (index, depth), linestyle in zip(dxw_labels.groupby('depth'), linestyles):
            xticks = depth['trainable_params']
            dxw_ax = ax.twiny()
            dxw_ax.set_xlim(ax.get_xlim())
            dxw_ax.set_xscale('log')
            dxw_ax.set_xticks(xticks)
            dxw_ax.set_xticklabels(depth['labels'], ha='left', rotation=45)
            dxw_ax.minorticks_off()
            # dxw_ax.set_xlabel('Depth x width')
            dxw_ax.grid(axis='x', linestyle=linestyle)
            for tick in dxw_ax.xaxis.get_majorticklabels():
                if tick.get_text() == '20x64':
                    tick.set_ha('center')
        # dxw_ax.set_xlabel('Depth x width')
        ax.grid(axis='both')
    ax.set_xlabel('# Parameters')
    ax.set_ylabel(ylabel if ylabel else y_key)
    if not show_arch:
        ax.grid(axis='x', which='both')
        ax.grid(axis='y')
        ax.minorticks_on()
    if legend_ax:
        h, l = ax.get_legend_handles_labels()
        legend_ax.legend(h, l, ncol=2)
    else:
        ax.legend()


def plot_param_arch_mnist(figsize, data, x_key, y_key, groupby, ylabel=None, label_fmt=None, colors=None,
                          markers=None, upper_interval=None, height_ratios=(3, 1), show_arch=True, architectures=None):
    sns.set_style('whitegrid')

    # If we were to simply plot pts, we'd lose most of the interesting
    # details due to the outliers. So let's 'break' or 'cut-out' the y-axis
    # into two portions - use the top (ax1) for the outliers, and the bottom
    # (ax2) for the details of the majority of our data
    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # fig.subplots_adjust(hspace=0.0)  # adjust space between axes
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=height_ratios)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    groups = [i for i in data.groupby(groupby)]
    # plot the same data on both axes
    for index, group in reversed(groups):
        group = group.sort_values('trainable_params')
        label = label_fmt(index) if label_fmt else index
        marker = markers[index[0]] if index[0] in markers else None
        ax1.scatter(group[x_key], group[y_key], label=label, alpha=0.5, color=colors[index], marker=marker)
        ax2.scatter(group[x_key], group[y_key], label=label, alpha=0.5, color=colors[index], marker=marker)

    # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(upper_interval[0], upper_interval[1])  # outliers only
    min_acc = data['val_acc'].min()
    margin = ((upper_interval[1] - upper_interval[0]) / (height_ratios[0] / height_ratios[1])) / 2
    print(margin)
    ax2.set_ylim(min_acc - margin, min_acc + margin)  # most of the data

    ax1.set_xscale('log')
    ax2.set_xscale('log')

    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='lightgray', mec='lightgray', mew=2, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    if show_arch:
        dxw_labels = data[[x_key, 'depth', 'width']].drop_duplicates().sort_values(x_key)
        dxw_labels['labels'] = dxw_labels[['depth', 'width']].astype(str).agg('x'.join, axis=1)
        if architectures:
            dxw_labels = dxw_labels[dxw_labels['labels'].isin(architectures)]

        xticks = dxw_labels['trainable_params']
        dxw_ax1 = ax1.twiny()
        dxw_ax1.set_xlim(ax1.get_xlim())
        dxw_ax1.set_xscale('log')
        dxw_ax1.set_xticks(xticks)
        dxw_ax1.set_xticklabels(dxw_labels['labels'], ha='center', rotation=45)
        dxw_ax1.minorticks_off()
        # dxw_ax.set_xlabel('Depth x width')
        dxw_ax1.grid(axis='y')

        dxw_ax1.set_xlabel('Depth x width')

        xticks = dxw_labels['trainable_params']
        dxw_ax2 = ax2.twiny()
        dxw_ax2.set_xlim(ax1.get_xlim())
        dxw_ax2.set_xscale('log')
        dxw_ax2.set_xticks(xticks)
        dxw_ax2.set_xticklabels([])
        dxw_ax2.minorticks_off()

        # dxw_ax.set_xlabel('Depth x width')
        dxw_ax2.grid(axis='y')

        dxw_ax1.set_xlabel('Depth x width')
        ax2.grid(axis='both')

    ax2.set_xlabel('# Parameters')
    ax1.set_ylabel(ylabel if ylabel else y_key)

    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    dxw_ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # dxw_ax2.spines.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    dxw_ax2.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    dxw_ax2.spines['top'].set_visible(False)
    ax1.grid(False, axis='both', which='both')
    ax1.minorticks_off()

    dxw_ax1.minorticks_off()

    # ax1.grid(axis='x', which='both')
    # ax1.grid(axis='y')
    # ax2.grid(axis='x', which='both')
    # ax2.grid(axis='y')
    # ax.minorticks_on()
    ax1.legend()
    return fig


def encode_params_str(params):
    params_str = []
    for k, v in params.items():
        letters = []
        words = [w for w in k.split('_') if len(w)]
        if len(words) > 1:
            for word in words:
                letters.append(word[0])
        else:
            letters.extend([words[0][0], words[0][-1]])

        letters = ''.join(letters)
        param_str = f'{letters}:{v}'
        params_str.append(param_str)
    return '_'.join(params_str)
