import torch
from torch.nn import functional as F
from torch.optim import Adam


class Trainer:

    def __init__(self, model, train_loader, eval_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.epoch = 0
        self.iter = 0
        self.criterion = F.nll_loss
        self.optimizer = Adam(model.parameters())

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), torch.tensor(target).to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            print("Train Epoch : {} [{} / {} {:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(self.train_loader), 100. * batch_idx / len(self.train_loader),
                loss.item()))
