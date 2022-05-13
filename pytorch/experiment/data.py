import glob
import os
import shutil
from pathlib import Path
from shutil import move
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torchvision
from accelerate import Accelerator
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
import os


dataset_shapes = {
    'cifar10': dict(input_shape=(3, 32, 32), output_shape=10),
    'simplenet-cifar10': dict(input_shape=(3, 32, 32), output_shape=10),
    'cifar10-toy': dict(input_shape=(3, 32, 32), output_shape=10),
    'mnist': dict(input_shape=(1, 28, 28), output_shape=10),
    'moons': dict(input_shape=(2,), output_shape=1),
    'blobs': dict(input_shape=(2,), output_shape=1),
    'imagenet': dict(input_shape=(3, 224, 224), output_shape=1000),
}


def load_data(dataset, batch_size=None, num_workers=0, n_samples=1000, random_state=None, noise=None,
              data_dir='./data'):
    data_dir = os.environ.get('DATA_DIR', data_dir)
    data_dir = Path(data_dir)

    # Create accelerator just for avoiding downlading everything k times
    accelerator = Accelerator()
    download = accelerator.is_local_main_process


    print(f'LOADING DATA FROM {data_dir}')

    if dataset == 'flattened_mnist':
        batch_size = batch_size if batch_size is not None else 1024

        train = pd.read_csv(data_dir / 'MNIST' / 'train.csv')
        # test = pd.read_csv(data_dir / 'MNIST' / 'test.csv')

        X_train = train.drop(columns=["label"])
        y_train = pd.get_dummies(train["label"])

        X_train, y_train = torch.Tensor(X_train.to_numpy()), torch.Tensor(y_train.to_numpy())
        # X_test = test.drop(columns=["label"])
        # y_test = y_train[:X_test.shape[0]]
        X_test = X_train[:10000]
        y_test = y_train[:10000]

        X_train = X_train / 255.0
        X_test = X_test / 255.0

        trainset = TensorDataset(X_train, y_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = TensorDataset(X_test, y_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        input_shape = (784,)
        output_shape = 10

    elif dataset == 'mnist':
        batch_size = batch_size if batch_size is not None else 1024

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        trainset = torchvision.datasets.MNIST(root=data_dir, train=True,
                                              download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root=data_dir, train=False,
                                             download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        input_shape = (1, 28, 28)
        output_shape = 10

    elif dataset == 'cifar10':
        batch_size = batch_size if batch_size is not None else 1024

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        input_shape = (3, 32, 32)
        output_shape = 10

    elif dataset == 'cifar100':
        batch_size = batch_size if batch_size is not None else 1024

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                                download=download, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
        input_shape = (3, 32, 32)
        output_shape = 100

    elif dataset == 'cifar10-toy':
        batch_size = batch_size if batch_size is not None else 1024

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                                download=download, transform=transform)

        train_idx, _ = train_test_split(
            np.arange(len(trainset)),
            train_size=n_samples,
            test_size=10,
            shuffle=True,
            stratify=trainset.targets,
            random_state=random_state)

        toyset = torch.utils.data.Subset(trainset, train_idx)
        trainloader = torch.utils.data.DataLoader(toyset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                               download=download, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        input_shape = (3, 32, 32)
        output_shape = 10

    elif dataset == 'moons':
        X, y = make_moons(n_samples=n_samples, shuffle=True, random_state=random_state, noise=noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
        X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float), \
                                           torch.tensor(X_test, dtype=torch.float), \
                                           torch.tensor(y_train, dtype=torch.float), \
                                           torch.tensor(y_test, dtype=torch.float)

        trainset = TensorDataset(X_train, y_train.unsqueeze(1))
        batch_size = batch_size if batch_size is not None else X_train.shape[0]

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = TensorDataset(X_test, y_test.unsqueeze(1))
        batch_size = batch_size if batch_size is not None else X_test.shape[0]

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        input_shape = (2,)
        output_shape = 1

    elif dataset == 'blobs':
        X, y = make_blobs(n_samples=n_samples, shuffle=True, random_state=random_state, centers=2, n_features=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
        X_train, X_test, y_train, y_test = torch.tensor(X_train, dtype=torch.float), \
                                           torch.tensor(X_test, dtype=torch.float), \
                                           torch.tensor(y_train, dtype=torch.float), \
                                           torch.tensor(y_test, dtype=torch.float)

        trainset = TensorDataset(X_train, y_train.unsqueeze(1))
        batch_size = batch_size if batch_size is not None else X_train.shape[0]

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=num_workers)

        testset = TensorDataset(X_test, y_test.unsqueeze(1))
        batch_size = batch_size if batch_size is not None else X_test.shape[0]

        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)

        input_shape = (2,)
        output_shape = 1

    else:
        raise ValueError(f'Unknown dataset {dataset}')

    return trainloader, testloader, input_shape, output_shape
