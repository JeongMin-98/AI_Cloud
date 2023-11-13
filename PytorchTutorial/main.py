import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from model import model
from model.model import check_device
from dataSet import datasetloader
from dataSet.data_transformer import get_transformations
from train.trainer import Trainer
from utils.tools import parse_model_config

if __name__ == "__main__":
    device = check_device()

    # import model from config file
    config = parse_model_config("./vggnet-19.cfg")

    model = model.MyCNN().to(device)
    summary(model, input_size=(3, 288, 288), batch_size=32, device=device)
    # custom_dataset = datasetloader.CustomImageDataSet(annotations_file="dataSet/imageSet/labels.csv",
    #                                                   img_dir="dataset/imageSet/")
    # data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
    #
    # train_size = int(0.8 * len(custom_dataset))
    # eval_size = len(custom_dataset) - train_size
    # train_dataset, eval_dataset = random_split(custom_dataset, [train_size, eval_size])
    #
    # train_dataset.dataset.transform = get_transformations()
    #
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    #
    # trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, device=device)
    # epochs = 10
    # trainer.train(epochs)
