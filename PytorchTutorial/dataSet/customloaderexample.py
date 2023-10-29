import numpy as np
from PIL import Image

from datasetloader import CustomImageDataSet
from showImage.utils import *

if __name__ == '__main__':
    dataset = CustomImageDataSet("imageSet/labels.csv", "./imageSet/")

    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        tensor2image(img_tensor)
