from datasetloader import CustomImageDataSet
from showImage.utils import *
from utils.tensor import get_attributes_of_tensor
from data_transformer import get_transformations

if __name__ == '__main__':
    dataset = CustomImageDataSet("imageSet/labels.csv", "./imageSet/", get_transformations((288,288)))

    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        print(dataset[i])
        get_attributes_of_tensor(img_tensor)
        # tensor2image(img_tensor)
        break