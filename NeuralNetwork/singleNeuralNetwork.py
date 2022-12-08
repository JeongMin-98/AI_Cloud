"""

    E-mail: jeongmin981@gmail.com
    author: 김정민(JeongMin Kim)
    Date: 2022.12.07

    Reference: [DKU]Machine Learning Theory / cs229 standford

    single neuron network

    Implement feed forward and backpropagation

"""

import numpy as np


class SingleNN:

    def __init__(self, inp):
        self.input_nodes = inp.shape[0]
        self.weight = None
        self.bias = None
        self.weighted_sum = None
        self.activation_func = self._relu

        self.weight = np.random.rand(self.input_nodes)
        self.bias = np.random.rand(self.input_nodes)

    def _relu(self, x):
        ret = np.where(x < 0, 0, x)
        return ret

    def derivate(self, x):
        if self.activation_func == self._relu:
            ret = np.where(x < 0, 0, 1)
            return ret

    def feed_forward(self, inp):
        self.weighted_sum = np.dot(self.weight.T, inp) + self.bias

        return self.activation_func(self.weighted_sum)

    def train(self, inp, y):
        output = self.feed_forward(inp)
        org = y

        error = output - org

        # Use Chain Rule
        # update weight and bias
        self.weight += error * self.derivate(output[-1]) * inp
        self.bias += error * self.derivate(output[-1])

    def fit(self, inp):
        return self.feed_forward(inp)


if __name__ == '__main__':
    data = np.array([1, 2, 3])
    y = np.array([2, 3, 4])
    nn = SingleNN(data)
    print(nn.weight)
    print(nn.bias)
    for _ in range(10):
        nn.train(data, y)
    print(nn.weight)
    print(nn.bias)

    print(nn.fit(data))
