import numpy as np
# import imgaug as ia
# from imgaug import augmenters as iaa
from torchvision import transforms as tf
from torchvision.transforms import InterpolationMode


def get_transformations(size=(288, 288)):
    """
    size만 조절하는 데이터 transformer 만들기
    향후 data augmentations을 위해서 추가
    :param size:
    :return:
    """
    data_transform = tf.Compose([
        # tf.Resize(size=(288, 288), interpolation=InterpolationMode.BILINEAR),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ResizeImage(new_size=size),
        ToFloatTensor(),
    ])

    return data_transform


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle
        self.rotation = tf.RandomRotation(angle)

    def __call__(self, data):
        image, label = data
        image = self.rotation(image)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self):
        self.transfer = tf.RandomHorizontalFlip()

    def __call__(self, data):
        image, label = data
        image = self.transfer(image)
        return image, label


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
        image, label = data
        image = image / 255
        # labels = torch.Tensor(np.array(labels, np.float32))

        return image, label


class ImgAug(object):
    """
        Image Augmentation Template
    """

    def __init__(self, augmentations=None):
        """
        Image Augmentation Initializer
        :param augmentations: type: iaa.Sequential
        """
        self.augmentations = augmentations

    def __call__(self, data):
        image, label = data
        # Convert tensor to numpy ndarray
        image = np.array(image)
        # apply image augmentation
        image = self.augmentations(image=image)

        return image, label


# Don't use Image Aug
# class DataAug(ImgAug):
#     def __init__(self):
#         super().__init__()
#         self.augmentations = iaa.Sequential([
#             iaa.Sharpen(0.0, 0.1),
#             iaa.Affine(rotate=(-0, 0),
#                        translate_percent=(-0.1, 0.1),
#                        scale=(0.8, 1.5)),
#         ])
