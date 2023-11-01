import os
import torch
from torch import nn

from ..Tensor.tensor import get_attributes_of_tensor


def check_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding_mode="zeros"),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layer(x)
        return x

if __name__ == "__main__":
    my_status_device = check_device()
    print(f"Using {my_status_device} device")

    model = MyCNN().to(my_status_device)
    print(model)

    input_feature_map = torch.randn(3,288,288)
    output = model.forward(x)
    get_attributes_of_tensor(output)

