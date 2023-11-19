from data_transformer import get_transformations
from datasetloader import CustomImageDataSet
from showImage.utils import *

if __name__ == '__main__':
    dataset = CustomImageDataSet("./imageSet2/labels.csv",
                                 "./imageSet2/",
                                 get_transformations((288, 288)))

    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        tensor2image(img_tensor)
