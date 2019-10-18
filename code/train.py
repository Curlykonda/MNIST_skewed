from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

from models import ConvNet


def accuracy(predictions, targets):
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
    accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()

    return accuracy


def train(model, optimizer, **kwargs):



