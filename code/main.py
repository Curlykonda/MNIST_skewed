#!/home/henning/PycharmProjects/MNIST_skewed/venv/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torchvision import datasets, transforms


import argparse
import os
import numpy as np
import pickle

import utils
from train import train_model
from models import ConvNet

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


def compose_res_dir(ARGS):

    res_dir = ARGS.res_dir

    res_dir += "/Exp_{:}_{:}".format(len(os.listdir(res_dir)) + 1, str(ARGS.skewed))
    if ARGS.weighted_loss:
        res_dir += "_weighted"
    else:
        res_dir += "_normal"

    if ARGS.data_aug:
        res_dir += "_DataAug"
    else:
        res_dir += "_NoAug"

    res_dir += "_{:}".format(ARGS.seed)

    os.mkdir(res_dir)

    return res_dir


def main():

    '''
    TODO:
        - Run archived Code on Colab GPU

    :return:
    '''
    print_args()

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)

    #get data
    mnist_trainset_normal = datasets.MNIST(train=True, root=ARGS.data_dir, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1307,), (0.3081,))
                                           ])
                                           )
    mnist_testset_normal = datasets.MNIST(root=ARGS.data_dir, train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ])
                                          )

    n_classes = len(mnist_testset_normal.classes)

    trainset = mnist_trainset_normal
    testset = mnist_testset_normal

    if ARGS.skewed:
        #TODO: modify trainset to only keep 3% of examples for classes 6 - 10
        trainset = mnist_testset_normal
        testset = mnist_testset_normal
        print("MNIST unbalanced")
    else:
        print("MNIST normal")

    print("Classes: {0} Train: {1} Test: {2}".format(n_classes, len(trainset), len(testset)))

    utils.show_class_distribution(trainset.targets, show=True)
    #print(trainset.data.shape)

    #set seeds
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)

    #initialise model
    convnet_model = ConvNet(in_channels=1, n_classes=10).to(device)
    print(convnet_model)
    optimizer = torch.optim.Adam(convnet_model.parameters(), lr=ARGS.learning_rate, )

    #create dataloader for train and test set
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=ARGS.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=ARGS.test_batch_size, shuffle=True, **kwargs)

    stats = {}
    #stats = train_model(convnet_model, optimizer, train_loader, test_loader, device, ARGS)

    #save results
    res_dir = compose_res_dir(ARGS)
    pickle.dump(stats, open(res_dir + "/stats.pkl", "wb"))

    #plot results
    utils.plot_train_test(res_dir)
    utils.plot_data_distribution(res_dir, trainset.targets)



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to train set.')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='Batch print(x.shape)size for test set')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval of printing training performance')
    parser.add_argument('--skewed', type=bool, default=False,
                        help='Indicate whether to keep dataset balanced or not')
    parser.add_argument('--weighted_loss', type=bool, default=False,
                        help='Use weighted loss to counter class imbalance')
    parser.add_argument('--data_aug', type=bool, default=False,
                        help='Use data augmentation to counter class imbalance')
    parser.add_argument('--data_dir', type=str, default="../data",
                        help='Directory for storing input data')
    parser.add_argument('--res_dir', type=str, default="../results",
                        help='Directory for storing results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    ARGS, unparsed = parser.parse_known_args()

    main()