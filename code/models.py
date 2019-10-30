from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
        nn.MaxPool2d(stride=2, **kwargs)
    )

class ConvNet(nn.Module):

    def __init__(self, in_channels, feature_maps=[32, 64], n_classes=10):

        super(ConvNet, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.feature_maps = [in_channels, *feature_maps]

        #create blocks
        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=0)
                       for in_f, out_f in zip(self.feature_maps, self.feature_maps[1:])]

        self.encoder = nn.Sequential(*conv_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=(feature_maps[-1]*4**2), out_features=256),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=n_classes)
        )

    def forward(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = x.view(x.size(0), -1) # flatten
        #print(x.shape)
        out = self.decoder(x.squeeze())
        #print(out.shape)

        return torch.nn.functional.log_softmax(out, dim=1)