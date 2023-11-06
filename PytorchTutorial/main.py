import torch
from torchsummary import summary
from model import model
from model.model import check_device
from dataSet import datasetloader
from dataSet.datasetloader import split_data
from dataSet.data_transformer import get_transformations
from train.trainer import Trainer

if __name__ == "__main__":
    device = check_device()
    model = model.MyCNN().to(device)
    # summary(model, input_size=(3, 288, 288), batch_size=1, device=device)
    dataloader = datasetloader.CustomImageDataSet(annotations_file="dataSet/imageSet2/labels.csv",
                                                  img_dir="dataset/imageSet2/",
                                                  transform=get_transformations(size=(288, 288)))

    train_loader, eval_loader = split_data(data=dataloader)

    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, device=device)
    epochs = 10
    trainer.train(epochs)
