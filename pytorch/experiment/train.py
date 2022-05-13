import random

import numpy as np
import torch
from accelerate import Accelerator
from torch import nn, optim
from torchsummary import summary

from experiment.metrics import multiclass_accuracy, binary_accuracy, compute_metrics
from jumpstart.loss import JumpstartRegularization


def train(model, trainloader, testloader, input_shape, output_shape, epochs, lr, lambda_, *, weight_decay=0,
          no_xent=False, aggr='norm', callbacks=None, axis=(0, 1), optimizer=None, seed=None,
          scheduler=None, balance=None, backprop_mode='single', split_batches=True, negative_margin=-1):
    if seed is not None:
        set_seed(seed)

    callbacks = [] if callbacks is None else callbacks
    accelerator = Accelerator(split_batches=split_batches)

    if output_shape > 1:
        criterion = nn.CrossEntropyLoss()
        accuracy = multiclass_accuracy
    else:
        criterion = nn.BCEWithLogitsLoss()
        accuracy = binary_accuracy

    if no_xent:
        def criterion(outputs, targets):
            return torch.tensor(0)

    if weight_decay > 0:
        model_parameters = add_weight_decay(model=model, weight_decay=weight_decay)
    else:
        model_parameters = model.parameters()

    if optimizer:
        optimizer = optimizer(model_parameters, lr=lr)
    else:
        optimizer = optim.Adam(model_parameters, lr=lr)
    if scheduler:
        scheduler = scheduler(optimizer)

    jumpstart = JumpstartRegularization(model=model, axis=axis, aggr=aggr, balance=balance, backprop_mode=backprop_mode,
                                       negative_margin=negative_margin)
    params = {'batch_size': trainloader.batch_size, 'lr': lr, 'sched': scheduler, 'optimizer': optimizer,
              'wd': weight_decay, 'lambda': lambda_, 'aggr': aggr, 'no_xent': no_xent}

    model, optimizer, trainloader, testloader = accelerator.prepare(model, optimizer, trainloader, testloader)
    accelerator.print('Training using the following params:')
    if accelerator.is_main_process:
        summary(model, input_shape)

    accelerator.print(f'epochs: {epochs} lr: {lr}, bs: {trainloader.batch_size}, lambda: {lambda_}, aggr: {aggr}')
    accelerator.print(
        f'Jumpstart params: aggr {jumpstart.aggr}, axis {jumpstart.axis}, balance {jumpstart.balance}, '
        f'n_params {len(jumpstart.parameter_losses)}')
    accelerator.print(f'opt: {optimizer}')
    if scheduler:
        accelerator.print(f'sched: {scheduler.__dict__}')

    logs = {'model': model, 'optimizer': optimizer, 'params': params}

    if accelerator.is_main_process:
        for c in callbacks:
            c.on_train_begin(logs)

    logs = {'model': accelerator.unwrap_model(model), 'optimizer': optimizer, 'epoch': 0, 'metric_data': {}}
    try:
        for epoch in range(epochs):
            model.train()
            if accelerator.is_main_process:
                logs = {'model': model, 'optimizer': optimizer, 'epoch': epoch}

                for c in callbacks:
                    c.on_epoch_begin(logs)

            # Metrics
            running_loss = 0.0
            running_sep = 0.0
            running_acc = 0.0
            running_samples = 0.0
            for batch, (inputs, targets) in enumerate(trainloader, 1):
                if accelerator.is_main_process:
                    logs = {'model': accelerator.unwrap_model(model), 'optimizer': optimizer, 'epoch': epoch,
                            'batch': batch}

                    for c in callbacks:
                        c.on_batch_begin(logs)

                # train
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                sep_loss = jumpstart.loss
                running_loss += loss.item()
                running_sep += sep_loss.item()

                if lambda_ > 0:
                    # Separation constriant
                    loss = loss + lambda_ * sep_loss

                accelerator.backward(loss)
                optimizer.step()

                # Evaluate model
                outputs, targets = accelerator.gather(outputs), accelerator.gather(targets)
                running_acc += accuracy(outputs, targets, aggregate=False).sum().item()
                running_samples += outputs.shape[0]
                if accelerator.is_main_process:
                    logs['metric_data'] = {'train_loss': running_loss / batch,
                                           'train_sep_loss': running_sep / batch,
                                           'train_lambda_sep_loss': running_sep * lambda_ / batch,
                                           'train_acc': running_acc / running_samples}

                    for c in callbacks:
                        c.on_batch_end(logs)

            model.eval()
            val_metrics = compute_metrics(model, criterion, testloader, lambda_, jumpstart, accuracy, accelerator)
            val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
            logs['metric_data'].update(val_metrics)
            if scheduler:
                logs['metric_data']['lr'] = scheduler.get_last_lr()
                scheduler.step()

            if accelerator.is_main_process:
                for c in callbacks:
                    c.on_epoch_end(logs)
    except StopIteration as ex:
        print(ex)

    if accelerator.is_main_process:
        for c in callbacks:
            c.on_train_end(logs)

    return logs


def add_weight_decay(model, weight_decay, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(0)
    random.seed(0)
