from datasetloader import CustomImageDataSet
from PIL import Image

if __name__ == '__main__':

    dataset = CustomImageDataSet("imageSet/labels.csv", "./imageSet/")

    # img plot
    img = Image.fromarray(dataset[0], 'RGB')
    img.save('my.png')
    # img.show()
    # for i in range(len(dataset)):
    #     img, label = dataset[i]
