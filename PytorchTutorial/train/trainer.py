import torch
from torch import nn
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
        self.running_loss = 0.
        self.running_acc = 0.
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.NLLLoss()
        # Define optimizer (Adam)
        self.optimizer = Adam(model.parameters(), lr=0.001)

    def save_model(self):
        path = "./pth/bestmodel.pth"
        torch.save(self.model.state_dict(), path)

    def test_accuracy(self):
        self.model.eval()
        accuracy = 0.
        total = 0.

        with torch.no_grad():
            for data in self.eval_loader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, -1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        accuracy = (100 * accuracy / total)
        return accuracy

    def train(self, num_epochs):
        # Print execution device
        print("The model will be running on ", self.device, "device")

        # declare the best accuracy
        best_accuracy = 0.0
        # In main.py Already Converted model parameters and buffers to Device.
        self.model.train()

        for epoch in range(num_epochs):  # loop over the dataset multimple times.

            self.running_loss = 0.
            self.running_acc = 0.

            for idx, (images, labels) in enumerate(self.train_loader):
                # get the inputs
                images = images.to(self.device)
                labels = torch.tensor(labels)
                # Convert labels to One-Hot Encoding
                labels_one_hot = F.one_hot(labels, num_classes=4).float()
                # labels_one_hot = labels_one_hot.view(-1).to(self.device)
                labels_one_hot = labels_one_hot.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # predict classes using images from the training set
                # output = self.model(images).view(-1)
                output = self.model(images)
                # compute the loss based on model output and real labels
                loss = self.loss_fn(output, labels_one_hot)
                # backpropagate ths loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                # Print statistics
                self.running_loss += loss.item()

                # if idx % 32 == 31:
                print('[%d, %5d] loss : %.4f' %
                      (epoch + 1, idx + 1, self.running_loss))
                self.running_loss = 0.

            # Compute and Print the average accuracy for this epoch
            accuracy = self.test_accuracy()
            print("For Epoch", epoch + 1, "the test accuracy over the test set is %d %%" % (accuracy))

            # if the accuracy is the best, Save the model.
            if accuracy > best_accuracy:
                self.save_model()
                best_accuracy = accuracy
