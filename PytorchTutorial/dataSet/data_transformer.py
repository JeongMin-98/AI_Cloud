import numpy as np
import torch
from torchvision import transforms as tf
from torchvision.transforms import InterpolationMode


def get_transformations(size=None):
    """
    size만 조절하는 데이터 transformer 만들기
    향후 data augmentations을 위해서 추가
    :param size:
    :return:
    """
    data_transform = tf.Compose([
        # tf.Resize(size=(288, 288), interpolation=InterpolationMode.BILINEAR),
        ResizeImage(),
        ToFloatTensor(),
    ])

    return data_transform


class ResizeImage(object):
    def __init__(self, new_size=(288, 288), interpolation=InterpolationMode.BILINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation
        self.resize = tf.Resize(self.new_size, self.interpolation)

    def __call__(self, data):
        image, label = data
        image = self.resize(image)
        return image, label


class ToFloatTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        image, labels = data
        image = image / 255
        # labels = torch.Tensor(np.array(labels, np.float32))

        return image, labels
