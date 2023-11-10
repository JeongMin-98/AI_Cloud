import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from utils.tools import check_device


class myVGG(nn.Module):

    def __init__(self, num_classes=4, info):
        super().__init__()
        self.info = info
        self.num_classes = num_classes
        self.conv_layer = _make_conv_layer(self.info)
        self.hiddlen_layer = nn.ReLU()
        self.fully_connected_layer = nn.Sequential([
            nn.Linear(in_features=512, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=self.num_classes)
        ])
        self.softmax = nn.softmax()

    def _make_conv_layer(self, info: dict):
        conv_layer = nn.Sequential()
        for key, item in info.items():
            # idx => 1, key [stride, padding, in_channels, out_channels]
            # max pooling layer => ~
            # conv_layer.append(conv2D(stride, padding, in_channels, out_channels))
            # conv_layer
            pass
