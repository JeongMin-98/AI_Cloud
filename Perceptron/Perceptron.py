import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
        self.activation_func = self._unit_step_func

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else -1 for i in y])

        for _ in range(self.n_iter):

            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weight) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr + (y_[idx] - y_predicted)

                self.weight += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weight) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x > 0, 1, -1)



