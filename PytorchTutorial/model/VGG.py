import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from utils.tools import check_device


class MyVgg(nn.Module):

    def __init__(self, num_classes=4, config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.conv_layer = self._make_conv_layer(self.config)
        self.fully_connected_layer = nn.Sequential([
            nn.Linear(in_features=512, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=self.num_classes),
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

        return None

    def set_layer(self):
        """ set layer from configuration file """
        module_list = nn.ModuleList()

        # Input size
        # pass

        for idx, info in enumerate(config):
            modules = nn.Sequential()

            if info['type'] == "convolutional":
                filters = int(info['filters'])
                modules = self._add_conv2d_layer(idx, modules, info, in_channels[-1], True)
                in_channels.append(in_channels[-1])
            elif info["type"] == "maxpool":
                modules.add_module("max pool", nn.MaxPool2d(kernel_size=info["size"], stride=info["stride"]))
