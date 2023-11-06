# Q1 강의 slide 15 에 있는 example 1 을 python 코드를
# 작성하여 실행 결과를 보이시오. (repeat 는 10 까지 한다)

import numpy as np

# SLP : Single Layer Perceptron
class SLP:
    def __init__(self, learning_rate=0.5, n_iter=10, func_name='constant'):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.input = None
        self.weight = None
        self.bias = None
        if func_name == "constant":
            self.activation_func = self._constant_func
        if func_name == "sigmoid":
            self.activation_func = self._sigmoid_func


    def fit(self, X, y, w):
        self.input = X
        self.weight = w
        self.bias = 0

        if self.activation_func == self._constant_func:
            print("현재 활성화 함수는 : constant")
        if self.activation_func == self._sigmoid_func:
            print("현재 활성화 함수는 : sigmoid")


        for i in range(self.n_iter):

            weighted_sum = np.dot(self.weight, self.input) + self.bias
            result = self.activation_func(weighted_sum)
            error = y - result

            print("{0}번째 weight : {1}".format(i+1, self.weight))
            print("error : {0:.3f}".format(error))

            for idx, x_i in enumerate(X):
                if self.activation_func == self._constant_func:
                    self.weight[idx] = self.weight[idx] + x_i * self.lr * error
                if self.activation_func == self._sigmoid_func:
                    self.weight[idx] = self.weight[idx] + x_i * self.lr * error * result * (1-result)

    def _constant_func(self, x):
        y = x
        return y

    def _sigmoid_func(self, x):
        return 1/(1+np.exp(-x))


input_value = np.array([0.5, 0.8, 0.2])
w = np.array([0.4, 0.7, 0.8])

a = SLP()
a.fit(input_value, 1, w)

a = SLP(n_iter=50, func_name="sigmoid")
a.fit(input_value, 1, w)


## simple delta rule

x = np.array([0.5, 0.8, 0.2])
w = np.array([0.4, 0.7, 0.8])
d = 1
alpha = 0.5


def _sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

for i in range(50):
    v = np.sum(w*x)
    y = _sigmoid_func(v)
    e = d - y
    print("error", i, e)
    print("=====================\n")
    print("weight : {}".format(w))
    w = w + alpha * y * (1-y) * e * x
