from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision

import argparse
import os
import numpy as np

#from models import ConvNet


def print_args():
    """
  Prints all entries in FLAGS variable.
  """
    for key, value in vars(ARGS).items():
        print(key + ' : ' + str(value))


def show_class_distribution(targets):
    #targets = [target for (_, target) in trainset]
    hist, bins = np.histogram(targets, bins=len(np.unique(targets)))

    for i in range(len(hist)):
        print("Digit: {0}  Train-Examples: {1} ({2} %)".format(i+1, hist[i], np.round(hist[i]/len(targets), decimals=3)*100))

    return hist




def main():

    print_args()

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    #get data
    mnist_trainset_normal = torchvision.datasets.MNIST(train=True, root=ARGS.data_dir, download=True, transform=None)
    mnist_testset_normal = torchvision.datasets.MNIST(root=ARGS.data_dir, train=False, download=True, transform=None)

    n_classes = len(mnist_testset_normal.classes)

    if ARGS.skewed:
        #TODO: modify trainset to only keep 3% of examples for classes 6 - 10
        trainset = mnist_testset_normal
        print("MNIST unbalanced")
    else:
        trainset = mnist_trainset_normal
        testset = mnist_testset_normal
        print("MNIST normal")

    print("Classes: {0} Train: {1} Test: {2}".format(n_classes, len(trainset), len(testset)))

    show_class_distribution(trainset.targets)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=1000,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default="../data",
                        help='Directory for storing input data')
    parser.add_argument('--skewed', type=bool, default=False,
                        help='Indicate whether to keep dataset balanced or not')

    ARGS, unparsed = parser.parse_known_args()



    main()