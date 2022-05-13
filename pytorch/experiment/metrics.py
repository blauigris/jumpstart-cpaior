import torch


def multiclass_accuracy(output, targets, aggregate=True):
    max_index = output.max(dim=1)[1]
    acc = (max_index == targets)
    return acc.mean() if aggregate else acc


def binary_accuracy(output, targets, aggregate=True):
    acc = ((output > 0.0) == targets).float()
    return acc.mean() if aggregate else acc


def compute_metrics(model, criterion, testloader, lambda_, shattercons, accuracy, accelerator):
    with torch.no_grad():
        running_accuracy = 0.0
        running_loss = 0.0
        running_sep = 0.0
        running_samples = 0.0

        for (inputs, targets) in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            running_sep += shattercons.loss.item()

            outputs, targets = accelerator.gather(outputs), accelerator.gather(targets)
            running_accuracy += accuracy(outputs, targets, aggregate=False).sum().item()
            running_samples += outputs.shape[0]
    return {'loss': running_loss / running_samples,
            'sep_loss': running_sep / running_samples,
            'lambda_sep_loss': (running_sep * lambda_) / running_samples,
            'acc': running_accuracy / running_samples,
            }