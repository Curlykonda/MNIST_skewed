from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

class ConvNet(nn.Module):

    def __init__(self, n_channels, feature_maps=[64, 128, 256], n_classes=10):

        super(ConvNet, self).__init__()

        #create blocks
        self.conv_layers == nn.ModuleList()

        self.conv_layers.add(*[
        nn.Conv2d(in_channels=n_channels, out_channels=64, stride=1, padding=1, kernel_size=(3, 3)),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        ])

        self.conv_layers.add(*[
            nn.Conv2d(in_channels=64, out_channels=128, stride=1, padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        ])

        self.conv_layers.add(*[
            nn.Conv2d(in_channels=128, out_channels=256, stride=1, padding=1, kernel_size=(3, 3)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        ])

        self.fc = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=n_classes)
        )

        def forward(self, x):
            print(x.shape)
            x = self.conv_layers(x)
            out = self.fc(x.squeeze())
            print(out.shape)

            return out
