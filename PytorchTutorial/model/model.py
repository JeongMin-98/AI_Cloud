import os
import torch
from torch import nn

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
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding_mode="zeros"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding_mode="zeros"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(start_dim=0),
            nn.Linear(in_features=73984, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=4),
            nn.Softmax(0),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == "__main__":
    my_status_device = check_device()
    print(f"Using {my_status_device} device")

    model = MyCNN().to(my_status_device)
    print(model)

    input_feature_map = torch.randn(3, 288, 288).to(my_status_device)
    output = model.forward(input_feature_map)
    print(f"Shape of tensor: {output.shape}")
    print(f"Datatype of tensor: {output.dtype}")
    print(f"Device tensor is stored on: {output.device}")

