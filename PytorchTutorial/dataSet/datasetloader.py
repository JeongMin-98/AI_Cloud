import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


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
