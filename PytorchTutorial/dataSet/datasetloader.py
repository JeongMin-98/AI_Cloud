import os
import pandas as pd
from torch.utils.data import dataloader
from torch.utils.data import Dataset
from torchvision.io import read_image
from sklearn.model_selection import train_test_split


def split_data(data, train_ratio=0.8, eval_ratio=0.2):
    train_data, eval_data = train_test_split(data, train_size=train_ratio, test_size=eval_ratio)
    train_loader = dataloader.DataLoader(train_data, batch_size=1, shuffle=True)
    eval_loader = dataloader.DataLoader(eval_data, batch_size=1, shuffle=True)
    return train_loader, eval_loader


class CustomImageDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        # check files Three channels
        if image.shape[0] != 3:
            return self[idx + 1]
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image, label = self.transform([image, label])
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
