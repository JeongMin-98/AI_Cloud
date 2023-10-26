"""

    E-mail: jeongmin981@gmail.com
    author: 김정민(JeongMin Kim)
    Date: 2022.12.07

    Reference: [DKU]Machine Learning Theory / cs229 standford

    Deep Neural Network
    Artificial neural networks consisting of multiple hidden layers between the input and output layers

    Method:
    __init__ => 초기 모델의 입력 feature와 출력 노드의 개수를 정한다.
    add_layer => 모델에 은닉층을 추가함
    train => 순전파와 역전파 과정을 통해 학습함.
    predict => 순전파와 역전파가 이루어진 모델에 데이터를 입력하여 결과를 출력함

"""

import numpy as np


class MultilayerNN:

    def __init__(self):
        self.layers = []
        self.weights = []
        self.bias = []

    def add_layer(self, nodes):
        if not self.layers:
            self.layers.append(nodes)
            return
        else:
            self.layers.append(nodes)
            self.weights.append(np.random.randn(size=(self.layers[-2], self.layers[-1])))
            self.bias.append(np.random.uniform(0, 0, size=(self.layers[-2], self.layers[-1])))


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        ret = np.where(x <= 0, 0, x)
        return ret

    def derivate(self, x):
        if self.activation_func == self._relu:
            ret = np.where(x <= 0, 0, 1)
            return ret
        if self.activation_func == self._sigmoid:
            return self._sigmoid(x) * (1 - self._sigmoid(x))

    def feed_forward(self, inp):
        self.weighted_sum = np.dot(self.weight.T, inp) + self.bias

        return self.activation_func(self.weighted_sum)

    def train(self, inp, y):

    def predict(self, inp):
