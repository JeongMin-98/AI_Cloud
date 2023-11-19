from torch import nn


def _add_hidden_layer(idx: int, modules, info: dict):
    if info['activation'] == 'relu':
        modules.add_module('layer_' + str(idx) + '_activation',
                           nn.ReLU())


class MyVgg(nn.Module):

    def __init__(self, config, num_classes=4):
        super().__init__()
        self.input_image_channel = 3
        self.config = config
        self.num_classes = num_classes
        self.conv_layer = self.set_layer()
        self.flatten_layer = nn.Flatten(start_dim=1)
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.softmax = nn.Softmax()

    def _add_conv2d_layer(self, idx: int, modules, info: dict, in_channel, batch_normalizations=True):

        # init Layer info
        filters = int(info['filters'])
        size = int(info['size'])
        stride = int(info['stride'])
        padding = int(info['padding'])

        modules.append(nn.Conv2d(in_channel,
                                 filters,
                                 size,
                                 stride,
                                 padding,
                                 ))

        if info['batch_normalize'] == '1':
            modules.append(nn.BatchNorm2d(filters))

        _add_hidden_layer(idx, modules, info)

        return modules

    def set_layer(self):
        """ set layer from configuration file """
        module_list = nn.ModuleList()
        in_channels = [self.input_image_channel]

        for idx, info in enumerate(self.config):
            modules = nn.Sequential()

            if info['type'] == "convolutional":
                filters = int(info['filters'])
                modules = self._add_conv2d_layer(idx, modules, info, in_channels[-1], True)
                in_channels.append(filters)
            elif info["type"] == "maxpool":
                modules.append(
                    nn.MaxPool2d(kernel_size=int(info["size"]), stride=int(info["stride"])))
            module_list.append(modules)
        return module_list

    def forward(self, x):

        # Forward Conv layer
        for idx, layer in enumerate(self.conv_layer):
            # print(idx, layer)
            x = layer(x)
            # print(x.shape)
        # print(x.shape)
        # Forward Fully Connected Layer
        x = self.flatten_layer(x)
        # print(x.shape)
        x = self.fully_connected_layer(x)

        # Forward SoftMax
        x = self.softmax(x)

        return x
