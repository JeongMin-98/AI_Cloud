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
        self.activation_func = self._sigmoid

        self.weight = np.random.randn(self.input_nodes)
        self.bias = np.random.randn(self.input_nodes)

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
        for _ in range(10):
            output = self.feed_forward(inp)
            org = y

            error = output - org

            # Use Chain Rule
            # update weight and bias
            self.weight -= error * self.derivate(output[-1]) * inp
            self.bias -= error * self.derivate(output[-1])
            print(self.weight)
            print(self.bias)

    def predict(self, inp):
        return self.feed_forward(inp)


if __name__ == '__main__':
    data = np.array([[0.1], [0.2], [0.3]])
    y = np.array([[0.9]])
    nn = SingleNN(data)
    print(nn.weight)
    print(nn.bias)

    nn.train(data, y)

    print(nn.predict(data))
