import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
"""
    dataloader +> use transformer
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    
    first layer => filter:16 kernel_size =(3, 3) activation relu
    layers.Conv2D(16, (3,3), padding='same', activation='relu'),
    MaxPooling2D default pool_size = (2, 2) 
    layers.MaxPooling2D(),
    The resulting output shape when using the "same" padding option is 
    output_shape = math.floor((input_shape - pool_size) / strides) + 1
    
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
"""


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Conv2d(3, )
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)

    # X = torch.rand(1, 28, 28, device=device)
    # logits = model(X)
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")
    #
    # input_image = torch.rand(3, 28, 28)
    # print(input_image.size())
    #
    # flatten = nn.Flatten()
    # flat_image = flatten(input_image)
    # print(flat_image.size())
    #
    # layer1 = nn.Linear(in_features=28 * 28, out_features=20)
    # hidden1 = layer1(flat_image)
    # print(hidden1.size())
    #
    # print(f"Before ReLU: {hidden1}\n\n")
    # hidden1 = nn.ReLU()(hidden1)
    # print(f"After ReLU: {hidden1}")
    #
    # seq_modules = nn.Sequential(
    #     flatten,
    #     layer1,
    #     nn.ReLU(),
    #     nn.Linear(20, 10)
    # )
    # input_image = torch.rand(3, 28, 28)
    # logits = seq_modules(input_image)
    #
    # softmax = nn.Softmax(dim=1)
    # pred_probab = softmax(logits)
    #
    # print(f"Model structure: {model}\n\n")
    #
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
