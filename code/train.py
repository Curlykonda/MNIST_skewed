from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def compute_accuracy(predictions, targets):
    """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  """
    # calculate accuracy over all predictions and average it
    # now we're dealing with tensors!
    accuracy = (predictions.argmax(dim=1) == targets).float().mean().item()

    return accuracy

def compute_f1_score(predictions, targets):
    '''
    precision := TP / TP + FP
    recall := TP / TP + FN
    f1 := 2 * P * R / P + R

    :param predictions:
    :param targets:
    :return:
    '''

    f1 = 0.0

    return f1


def train_model(model, optimizer, train_loader, test_loader, device, args):

    loss_fn = nn.CrossEntropyLoss()

    stats = {}
    stats["train_acc"] = []
    stats["train_f1"] = []
    stats["train_loss"] = []
    stats["test_acc"] = []
    stats["test_loss"] = []
    stats["test_f1"] = []

    #format data correctly
    #assert data_shape == model.in_channels

    model.train()

    for epoch in range(1, args.epochs+1):

        train_loss = []
        train_acc = []
        train_f1 = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)    #perform forward pass
            loss = nn.functional.nll_loss(output, targets) #compute loss

            # perform  backward pass
            loss.backward()
            optimizer.step()

            #log stats
            train_loss.append(loss.item())
            train_acc.append(accuracy_score(output.argmax(dim=1), targets)) # sklearn.metrics
            train_f1.append(f1_score(output.argmax(dim=1), targets, average='weighted'))

            if batch_idx % args.log_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Acc: {:.1f} \t F1: {:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), train_acc[-1]*100, train_f1[-1]))

        stats["train_acc"].append(np.mean(train_acc))
        stats["train_f1"].append(np.mean(train_f1))
        stats["train_loss"].append(np.mean(train_loss))

        # evaluate on test set
        test_acc, test_f1, test_loss = test_model(model, test_loader, device)

        stats["test_acc"].append(np.mean(test_acc))
        stats["test_f1"].append(np.mean(test_f1))
        stats["test_loss"].append(np.mean(test_loss))

        print('\n Evaluation on test set:')
        print('Loss: {:.6f} \t Acc: {:.1f} \t F1: {:.4f} \n'.format(
            stats["test_loss"][-1], stats["test_acc"][-1] * 100, stats["test_f1"][-1]))

    return stats

def test_model(model, test_loader, device):
    model.eval()
    test_acc, test_f1, test_loss = [], [], []

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            loss = nn.functional.nll_loss(output, targets) #compute loss

            test_acc.append(accuracy_score(output.argmax(dim=1), targets))  # sklearn.metrics
            test_f1.append(f1_score(output.argmax(dim=1), targets, average='weighted'))
            test_loss.append(loss.item())

    return test_acc, test_f1, test_loss

