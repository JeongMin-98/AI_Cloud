import torch
from torch import nn
from torch.nn import functional as F

from utils.tensor import get_attributes_of_tensor
from utils.tools import check_device


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_augmentation = nn.Sequential(

        )
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=73984, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4),
            # nn.Softmax(0),
        )

    def forward(self, x):
        x = self.layer(x)
        x = F.softmax(x, dim=1)
        # x = torch.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    my_status_device = check_device()
    print(f"Using {my_status_device} device")

    model = MyCNN().to(my_status_device)
    # criterion = F.nll_loss
    # optimizer = Adam(model.parameters())
    print(model)

    input_feature_map = torch.randn(3, 288, 288).to(my_status_device)
    output = model.forward(input_feature_map)
    get_attributes_of_tensor(output)
    output.to("cpu")
    print(output)
