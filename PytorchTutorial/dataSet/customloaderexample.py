from datasetloader import CustomImageDataSet
from showImage.utils import *
from utils.tensor import get_attributes_of_tensor
from data_transformer import get_transformations

if __name__ == '__main__':
    dataset = CustomImageDataSet("./imageSet2/labels.csv",
                                 "./imageSet2/",
                                 get_transformations((288, 288)))


    for i in range(len(dataset)):
        print(i)
        img_tensor, label = dataset[i]
        # tensor2image(img_tensor)
