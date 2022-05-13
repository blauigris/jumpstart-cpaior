import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

from experiment.experiment import extract_grids
from experiment.plotting import export_grid_plots, plot_param_arch, plot_param_arch_mnist

os.environ['WANDB_MODE'] = 'offline'


def plot_cifar_param_count():
    sns.set_context('talk', font_scale=1.3)
    figsize = (15, 8)
    cifar = pd.read_csv('cifar10.csv')
    cifar['time'] = pd.to_timedelta(cifar['time'])
    cifar = cifar[(cifar['mode'] == 'UP') | (cifar['mode'].isna())]
    cifar = cifar[~cifar['batchnorm']]
    cifar = cifar[cifar['negative_margin'] == -1]
    cifar = cifar[cifar['init'] == 'kaiming']
    cifar = cifar[cifar['kernel_size'] == '(3, 3)']
    cifar = cifar[cifar['n_maxpool'] == 0]
    cifar.loc[cifar['lambda'] == 0, 'aggr'] = 'None'
    cifar = cifar[(cifar['width'] <= 192) & (cifar['width'] != 128)]

    sns.set_style('whitegrid')

    groupby = ['lr', 'aggr', 'lambda', ]

    def label_fmt(x):
        if x[1] == 'None':
            label = f'Baseline $\lambda = 0, \epsilon = {x[0]}$'
        else:
            if x[1] == 'mean':
                aggr = r'\bar{x}'
            elif x[1] == 'norm':
                aggr = r'L^2'
            else:
                raise ValueError(f'Unknown aggr {x[1]}')
            label = f'${aggr}, \lambda = {x[2]}, \epsilon = {x[0]}$'
        return label

    blues = sns.color_palette("Blues", as_cmap=True)
    greens = sns.color_palette("Greens", as_cmap=True)
    colors = {('None', 0.0): 'red',
              ('mean', 0.1): blues(0.5),
              ('mean', 1): blues(0.9),
              ('norm', 0.001): greens(0.5),
              ('norm', 0.1): greens(0.9)
              }
    markers = {('None', 0.001): "p",
               ('None', 0.0001): 's',
               ('mean', 0.001): "^",
               ('mean', 0.0001): "v",
               ('norm', 0.001): "^",
               ('norm', 0.0001): "v",
               }
    architectures = ['10x2', '30x2', '10x16', '30x15', '10x64', '30x64', '10x192', '30x192',
                     '20x2', '20x8', '20x16', '20x32', '20x64', '20x96', '20x192']

    fig, ax = plt.subplots(1, figsize=figsize)
    legend_fig = plt.figure(figsize=(10.4, 2.5))
    plot_param_arch(ax=ax, data=cifar, x_key='trainable_params', y_key='train_acc',
                    ylabel='Training accuracy',
                    groupby=groupby,
                    label_fmt=label_fmt,
                    colors=colors,
                    markers=markers,
                    architectures=architectures,
                    legend_ax=legend_fig)
    fig.tight_layout()
    fig.savefig(f'../plots/cifar_param_count-train.pdf')
    legend_fig.savefig(f'../plots/cifar_param_count_legend-train.pdf')

    plt.close(fig)
    plt.close(legend_fig)

    fig, ax = plt.subplots(1, figsize=figsize)
    legend_fig = plt.figure(figsize=(10.4, 2.5))
    plot_param_arch(ax=ax, data=cifar, x_key='trainable_params', y_key='val_acc',
                    ylabel='Validation accuracy',
                    groupby=groupby,
                    label_fmt=label_fmt,
                    colors=colors,
                    markers=markers, architectures=architectures,
                    legend_ax=legend_fig)
    fig.tight_layout()
    fig.savefig(f'../plots/cifar_param_count-val.pdf')
    legend_fig.savefig(f'../plots/cifar_param_count_legend-val.pdf')

    plt.close(fig)
    plt.close(legend_fig)


def plot_cifar_grid():
    figsize = (2, 4)
    cbar_figsize = (1, 4)
    cifar = pd.read_csv('cifar10.csv')
    cifar['time'] = pd.to_timedelta(cifar['time'])
    cifar = cifar[(cifar['mode'] == 'UP') | (cifar['mode'].isna())]
    cifar = cifar[~cifar['batchnorm']]
    cifar = cifar[cifar['negative_margin'] == -1]
    cifar = cifar[cifar['init'] == 'kaiming']
    cifar = cifar[cifar['kernel_size'] == '(3, 3)']
    cifar = cifar[cifar['n_maxpool'] == 0]
    # cifar = cifar[cifar['lr'] == 0.001]
    # cifar = cifar[cifar['aggr'] == 'norm']
    cifar.loc[cifar['lambda'] == 0, 'aggr'] = None

    cifar = cifar[(cifar['width'] <= 192) & (cifar['width'] != 128)]
    # cifar = cifar[cifar['depth'] <= 96]
    hparams = cifar.loc[:, :'filename'].columns.to_list()
    cifar = cifar.set_index(hparams + ['epochs'])
    cifar['alive_point_ratio'] = 1 - cifar['dead_point_ratio']
    cifar['init_alive_point_ratio'] = 1 - cifar['init_dead_point_ratio']
    cifar = cifar[['train_acc',
                   'val_acc',
                   'init_separating_unit_ratio',
                   'separating_unit_ratio',
                   'init_alive_point_ratio',
                   'alive_point_ratio']]

    grids = extract_grids(cifar)
    hparams = grids.index.names
    grids = grids.reset_index()
    grids = grids.sort_values(['lr', 'aggr', 'lambda'])
    grids = grids.set_index(hparams)

    export_grid_plots(grids, Path('../plots/cifar_grid'), figsize=figsize, vmin=0, vmax=1,
                      fmt='.2g', latex_path_prefix='images/cifar10_grid', width=0.32, annot=True,
                      cbar_figsize=cbar_figsize)


def download_data(entity='blauigris', project='cifar100', dataset='cifar100'):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        history = run.scan_history()
        summaries = pd.DataFrame([row for row in history])
        summary = {}
        for metric in summaries:
            if metric.endswith('acc'):
                best = summaries[metric].max()
                summary[metric] = best
            elif metric.endswith('loss'):
                best = summaries[metric].min()
                summary[metric] = best

        pd.Series(summary)

        summary_list.append(summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k, v in run.config.items()
                  if not k.startswith('_')}
        config['dataset'] = dataset
        config['optimizer'] = 'adam'
        config['epochs'] = 400
        config_list.append(config)

        # .name is the human-readable name of the run.
        name_list.append(run.name)
    runs_df = pd.concat((pd.DataFrame(config_list), pd.DataFrame(summary_list)), keys=['hparams', 'metrics'],
                        axis=1)
    runs_df.set_index(pd.Series(name_list), inplace=True)
    runs_df.to_csv(f"{project}.csv")


def plot_cifar100_grid():
    figsize = (2, 4)
    cbar_figsize = (1, 4)
    cifar100 = pd.read_csv('cifar100.csv', index_col=0, header=[0, 1])

    # imagewoof160 = imagewoof160[imagewoof160['depth'] <= 96]
    cifar100 = cifar100.droplevel(0, axis=1).set_index(cifar100['hparams'].columns.tolist())

    cifar100 = cifar100[['train_acc', 'val_acc']]

    grids = extract_grids(cifar100)
    hparams = grids.index.names
    grids = grids.reset_index()
    grids = grids.sort_values(['lr', 'aggr', 'lambda'])
    grids = grids.set_index(hparams)

    export_grid_plots(grids, Path('../plots/cifar100'), figsize=figsize, vmin=0, vmax=1,
                      fmt='.2g', latex_path_prefix='images/cifar100', width=0.32, annot=True,
                      cbar_figsize=cbar_figsize, format='jpeg')


def plot_cifar_grid_table():
    figsize = (5, 10)
    cifar = pd.read_csv('cifar10.csv')
    cifar['time'] = pd.to_timedelta(cifar['time'])
    cifar = cifar[(cifar['mode'] == 'UP') | (cifar['mode'].isna())]
    cifar = cifar[~cifar['batchnorm']]
    cifar = cifar[cifar['negative_margin'] == -1]
    cifar = cifar[cifar['init'] == 'kaiming']
    cifar = cifar[cifar['kernel_size'] == '(3, 3)']
    cifar = cifar[cifar['n_maxpool'] == 0]
    cifar = cifar[cifar['lr'] == 0.001]
    # cifar = cifar[cifar['aggr'] == 'norm']
    cifar.loc[cifar['lambda'] == 0, 'aggr'] = 'None'
    # cifar.loc[cifar['lambda'] > 0, 'aggr'] = 'RC'

    cifar = cifar[(cifar['width'] <= 192) & (cifar['width'] != 128)]
    # cifar = cifar[cifar['depth'] <= 96]
    hparams = cifar.loc[:, :'filename'].columns.to_list()
    cifar = cifar.set_index(hparams + ['epochs'])
    cifar = cifar[['train_acc', 'val_acc']]
    acc = cifar.groupby(['aggr', 'lambda']).max()
    acc.to_latex('../plots/cifar_grid_table.tex')


def plot_moon_grid():
    sns.set_context('talk', font_scale=1.5)
    # sns.set_context('talk', font_scale=2.35)
    figsize = (7, 3.5)
    cbar_figsize = (1.75, 5)
    moons = pd.read_csv('moons.csv')
    moons['time'] = pd.to_timedelta(moons['time'])

    # moons = moons[(moons['mode'] == 'UP') | (moons['mode'].isna())]
    moons = moons[~moons['batchnorm']]
    # moons = moons[moons['negative_margin'] == -1]
    # moons = moons[moons['init'] == 'glorot']
    moons['epochs'] = 5000

    moons = moons[moons['width'] <= 25]
    # moons = moons[moons['depth'] <= 60]
    hparams = moons.loc[:, :'filename'].columns
    moons = moons.set_index(hparams.to_list() + ['epochs'])
    moons = moons[['train_acc', 'val_acc']]

    grids = extract_grids(moons)

    export_grid_plots(grids, Path('../plots/moons_grid'), figsize=figsize, annot=False, cbar_figsize=cbar_figsize,
                      vmax=1, vmin=0.51, xticklabels='auto', yticklabels=4)


def plot_mnist_grid():
    figsize = (10, 5)
    cbar_figsize = (1, 5)
    mnist = pd.read_csv('mnist.csv', index_col=0)
    mnist['time'] = pd.to_timedelta(mnist['time'])
    # TODO FIX lambda export 1e-8
    mnist.loc[(mnist['mode'] == 'UP') & (mnist['kernel_size'] == '(3, 3)'), 'lambda'] = 1e-8
    mnist['lambda'] = pd.to_numeric(mnist['lambda'])
    mnist = mnist[(mnist['mode'] == 'UP') | (mnist['mode'].isna())]
    mnist = mnist[~mnist['batchnorm']]
    mnist = mnist[mnist['negative_margin'] == -1]
    mnist = mnist[mnist['init'] == 'glorot']
    mnist = mnist[mnist['kernel_size'] == '(3, 3)']
    mnist['epochs'] = 50

    mnist = mnist[mnist['width'] <= 8]
    # mnist = mnist[mnist['depth'] <= 60]

    hparams = mnist.loc[:, :'filename'].columns
    mnist = mnist.set_index(hparams.to_list() + ['epochs'])
    mnist = mnist[['train_acc', 'val_acc']]

    grids = extract_grids(mnist)

    export_grid_plots(grids, Path('../plots/mnist_grid'), figsize=figsize, annot=True, cbar_figsize=cbar_figsize,
                      vmin=0.1, vmax=1, fmt='.3g')


def plot_mnist_param_count():
    sns.set_context('talk', font_scale=1.3)
    figsize = (15, 10)
    mnist = pd.read_csv('mnist.csv', index_col=0)
    mnist['time'] = pd.to_timedelta(mnist['time'])
    # TODO FIX lambda export 1e-8
    mnist.loc[(mnist['mode'] == 'UP') & (mnist['kernel_size'] == '(3, 3)'), 'lambda'] = 1e-8
    mnist['lambda'] = pd.to_numeric(mnist['lambda'])
    mnist = mnist[(mnist['mode'] == 'UP') | (mnist['mode'].isna())]
    mnist = mnist[~mnist['batchnorm']]
    mnist = mnist[mnist['negative_margin'] == -1]
    mnist = mnist[mnist['init'] == 'glorot']
    mnist = mnist[mnist['kernel_size'] == '(3, 3)']
    mnist['epochs'] = 50
    mnist.loc[mnist['lambda'] == 0, 'aggr'] = 'None'
    mnist.loc[(mnist['mode'].isna()), 'mode'] = 'None'

    mnist = mnist[mnist['width'] <= 8]
    # mnist = mnist[mnist['depth'] <= 60]

    sns.set_style('whitegrid')

    groupby = ['mode', 'lambda']

    def label_fmt(x):
        if x[0] == 'None':
            label = 'Baseline'
        else:
            label = f'{x[0]} $\lambda = {x[1]}$'
        return label

    blues = sns.color_palette("Blues", as_cmap=True)
    greens = sns.color_palette("Greens", as_cmap=True)
    colors = {('None', 0.0): 'red',
              ('UP', 1e-8): blues(0.9),
              }
    markers = {'None': "s"}

    architectures = ['1x2',
                     # '2x2',
                     # '1x4',
                     # '4x2',
                     # '2x4',
                     '8x2',
                     # '1x8',
                     # '12x2',
                     # '4x4',
                     # '16x2',
                     # '20x2',
                     '2x8',
                     # '24x2',
                     # '28x2',
                     # '8x4',
                     # '32x2',
                     # '36x2',
                     # '40x2',
                     # '44x2',
                     # '12x4',
                     # '48x2',
                     # '52x2',
                     # '4x8',
                     # '56x2',
                     # '60x2',
                     # '16x4',
                     # '64x2',
                     '68x2',
                     # '20x4',
                     # '24x4',
                     # '28x4',
                     # '8x8',
                     # '32x4',
                     # '36x4',
                     # '40x4',
                     # '44x4',
                     '12x8',
                     # '48x4',
                     # '52x4',
                     # '56x4',
                     # '60x4',
                     # '16x8',
                     # '64x4',
                     # '68x4',
                     # '20x8',
                     # '24x8',
                     # '28x8',
                     # '32x8',
                     # '36x8',
                     '40x8',
                     # '44x8',
                     # '48x8',
                     # '52x8',
                     # '56x8',
                     # '60x8',
                     # '64x8',
                     '68x8'
                     ]

    fig = plot_param_arch_mnist(figsize, data=mnist, x_key='trainable_params', y_key='train_acc',
                                ylabel='Train accuracy',
                                groupby=groupby,
                                label_fmt=label_fmt,
                                colors=colors,
                                markers=markers,
                                upper_interval=[0.965, 1.001],
                                show_arch=True,
                                architectures=architectures)
    fig.tight_layout()

    fig.savefig(f'../plots/mnist_param_count-train.pdf')
    plt.close(fig)

    fig = plot_param_arch_mnist(figsize, data=mnist, x_key='trainable_params', y_key='val_acc',
                                ylabel='Validation accuracy',
                                groupby=groupby,
                                label_fmt=label_fmt,
                                colors=colors,
                                markers=markers,
                                upper_interval=[0.994 - (1.001 - 0.965), 0.994],
                                architectures=architectures
                                )
    fig.tight_layout()
    fig.savefig(f'../plots/mnist_param_count-val.pdf')
    plt.close(fig)


def plot_mnist_times_better():
    mnist = pd.read_csv('mnist.csv', index_col=0)
    mnist['time'] = pd.to_timedelta(mnist['time'])
    # TODO FIX lambda export 1e-8
    mnist.loc[(mnist['mode'] == 'UP') & (mnist['kernel_size'] == '(3, 3)'), 'lambda'] = 1e-8
    mnist['lambda'] = pd.to_numeric(mnist['lambda'])
    mnist = mnist[(mnist['mode'] == 'UP') | (mnist['mode'].isna())]
    mnist = mnist[~mnist['batchnorm']]
    mnist = mnist[mnist['negative_margin'] == -1]
    mnist = mnist[mnist['init'] == 'glorot']
    mnist = mnist[mnist['kernel_size'] == '(3, 3)']
    mnist['epochs'] = 50
    mnist.loc[mnist['lambda'] == 0, 'aggr'] = 'None'
    mnist.loc[(mnist['mode'].isna()), 'mode'] = 'None'

    mnist = mnist[mnist['width'] <= 8]
    mnist = mnist[mnist['depth'] > 1]
    train_acc = mnist[['depth', 'width', 'mode', 'train_acc']]
    train_acc = train_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'train_acc'])  # Drop ties
    best = train_acc.loc[train_acc.groupby(['depth', 'width'])['train_acc'].idxmax()]
    train_acc_counts = best.value_counts('mode')
    print('TRAIN')
    print(train_acc_counts)

    val_acc = mnist[['depth', 'width', 'mode', 'val_acc']]
    val_acc = val_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'val_acc'])  # Drop ties
    best = val_acc.loc[val_acc.groupby(['depth', 'width'])['val_acc'].idxmax()]
    val_acc_counts = best.value_counts('mode')
    print('VALIDATION')
    print(val_acc_counts)


def plot_mnist_grid_table():
    mnist = pd.read_csv('mnist.csv', index_col=0)
    mnist['time'] = pd.to_timedelta(mnist['time'])
    # TODO FIX lambda export 1e-8
    mnist.loc[(mnist['mode'] == 'UP') & (mnist['kernel_size'] == '(3, 3)'), 'lambda'] = 1e-8
    mnist['lambda'] = pd.to_numeric(mnist['lambda'])
    mnist = mnist[(mnist['mode'] == 'UP') | (mnist['mode'].isna())]
    mnist = mnist[~mnist['batchnorm']]
    mnist = mnist[mnist['negative_margin'] == -1]
    mnist = mnist[mnist['init'] == 'glorot']
    mnist = mnist[mnist['kernel_size'] == '(3, 3)']
    mnist['epochs'] = 50
    mnist.loc[mnist['lambda'] == 0, 'aggr'] = 'None'
    mnist.loc[(mnist['mode'].isna()), 'mode'] = 'None'

    mnist = mnist[mnist['width'] <= 8]
    mnist = mnist[mnist['depth'] > 1]

    # cifar = cifar[cifar['depth'] <= 96]
    hparams = mnist.loc[:, :'filename'].columns.to_list()
    mnist = mnist.set_index(hparams + ['epochs'])
    mnist = mnist[['train_acc', 'val_acc']]
    acc = mnist.groupby(['mode']).max()
    acc.to_latex('../plots/mnist_grid_table.tex')


def plot_moons_grid_table():
    sns.set_context('talk', font_scale=1.5)
    figsize = (7, 3.5)
    cbar_figsize = (1.75, 5)
    moons = pd.read_csv('moons.csv')
    moons['time'] = pd.to_timedelta(moons['time'])

    # moons = moons[(moons['mode'] == 'UP') | (moons['mode'].isna())]
    moons = moons[~moons['batchnorm']]
    # moons = moons[moons['negative_margin'] == -1]
    moons = moons[moons['init'] == 'glorot']
    moons['epochs'] = 5000

    moons = moons[moons['width'] <= 25]

    # cifar = cifar[cifar['depth'] <= 96]
    hparams = moons.loc[:, :'filename'].columns.to_list()
    moons = moons.set_index(hparams + ['epochs'])
    moons = moons[['train_acc', 'val_acc']]
    acc = moons.groupby(['mode']).max()
    acc.to_latex('../plots/moons_grid_table.tex')


def plot_cifar_times_better():
    sns.set_context('talk', font_scale=1.3)
    figsize = (15, 10)
    cifar = pd.read_csv('cifar10.csv')
    cifar['time'] = pd.to_timedelta(cifar['time'])
    cifar = cifar[(cifar['mode'] == 'UP') | (cifar['mode'].isna())]
    cifar = cifar[~cifar['batchnorm']]
    cifar = cifar[cifar['negative_margin'] == -1]
    cifar = cifar[cifar['init'] == 'kaiming']
    cifar = cifar[cifar['kernel_size'] == '(3, 3)']
    cifar = cifar[cifar['n_maxpool'] == 0]
    cifar.loc[cifar['lambda'] == 0, 'aggr'] = 'None'
    # cifar.loc[cifar['lambda'] > 0, 'aggr'] = 'JR'
    cifar = cifar[(cifar['width'] <= 192) & (cifar['width'] != 128)]

    train_acc = cifar[['depth', 'width', 'aggr', 'lr', 'lambda', 'train_acc']]
    # train_acc = train_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'train_acc'])  # Drop ties
    best = train_acc.loc[train_acc.groupby(['depth', 'width', 'lr'])['train_acc'].idxmax()]
    train_acc_counts = best.value_counts(['aggr', 'lambda', 'lr']).sort_index()
    print('TRAIN')
    print(train_acc_counts.to_latex())
    train_acc_counts.to_latex('cifar_times_better-train.tex')

    val_acc = cifar[['depth', 'width', 'aggr', 'lr', 'lambda', 'val_acc']]
    # val_acc = val_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'val_acc'])  # Drop ties
    best = val_acc.loc[val_acc.groupby(['depth', 'width', 'lr'])['val_acc'].idxmax()]
    val_acc_counts = best.value_counts(['aggr', 'lambda', 'lr']).sort_index()
    val_acc_counts.to_latex('cifar_times_better-val.tex')

    print('VALIDATION')
    print(val_acc_counts.to_latex())


def plot_cifar_times_better_baseline():
    sns.set_context('talk', font_scale=1.3)
    figsize = (15, 10)
    cifar = pd.read_csv('cifar10.csv')
    cifar['time'] = pd.to_timedelta(cifar['time'])
    cifar = cifar[(cifar['mode'] == 'UP') | (cifar['mode'].isna())]
    cifar = cifar[~cifar['batchnorm']]
    cifar = cifar[cifar['negative_margin'] == -1]
    cifar = cifar[cifar['init'] == 'kaiming']
    cifar = cifar[cifar['kernel_size'] == '(3, 3)']
    cifar = cifar[cifar['n_maxpool'] == 0]
    cifar.loc[cifar['lambda'] == 0, 'aggr'] = 'None'
    # cifar.loc[cifar['lambda'] > 0, 'aggr'] = 'JR'
    cifar = cifar[(cifar['width'] <= 192) & (cifar['width'] != 128)]

    train_acc = cifar[['depth', 'width', 'aggr', 'lr', 'lambda', 'train_acc']]
    # train_acc = train_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'train_acc'])  # Drop tie
    wins = defaultdict(lambda: 0)
    for index, group in train_acc.groupby(['depth', 'width', 'lr']):
        group.drop(columns=['depth', 'width'], inplace=True)
        baseline = group[group['aggr'] == 'None']
        jumpstarts = group[group['aggr'] != 'None']
        win = jumpstarts['train_acc'].values > baseline['train_acc'].values
        jumpstarts['wins'] = win.astype(int)
        jumpstarts.set_index(['aggr', 'lambda', 'lr'], inplace=True)
        jumpstarts = jumpstarts['wins']
        for j_index, won in jumpstarts.iteritems():
            wins[j_index] += won

    train_counts = pd.DataFrame.from_dict(wins, orient='index')
    print('TRAIN')
    print(train_counts.to_latex())
    train_counts.to_latex('cifar_times_better_baselin-train.tex')

    val_acc = cifar[['depth', 'width', 'aggr', 'lr', 'lambda', 'val_acc']]
    # val_acc = val_acc.drop_duplicates(keep=False, subset=['depth', 'width', 'val_acc'])  # Drop tie
    wins = defaultdict(lambda: 0)
    for index, group in val_acc.groupby(['depth', 'width', 'lr']):
        group.drop(columns=['depth', 'width'], inplace=True)
        baseline = group[group['aggr'] == 'None']
        jumpstarts = group[group['aggr'] != 'None']
        win = jumpstarts['val_acc'].values > baseline['val_acc'].values
        jumpstarts['wins'] = win.astype(int)
        jumpstarts.set_index(['aggr', 'lambda', 'lr'], inplace=True)
        jumpstarts = jumpstarts['wins']
        for j_index, won in jumpstarts.iteritems():
            wins[j_index] += won

    val_counts = pd.DataFrame.from_dict(wins, orient='index')
    print('VAL')
    print(val_counts.to_latex())
    val_counts.to_latex('cifar_times_better_baselin-val.tex')
