import torch
from torchsummary import summary
from model import model
from model.model import check_device
from dataSet import datasetloader
from train.trainer import Trainer

if __name__ == "__main__":
    device = check_device()
    model = model.MyCNN().to(device)
    # summary(model, input_size=(3, 288, 288), batch_size=-1, device=device)
    dataloader = datasetloader.CustomImageDataSet("dataSet/imageSet/labels.csv",
                                                  "dataSet/imageSet/")

    trainer = Trainer(model=model, train_loader=dataloader, eval_loader=None, device=device)
    epochs = 10
    for epoch in range(1, epochs + 1):
        trainer.train(epoch)
