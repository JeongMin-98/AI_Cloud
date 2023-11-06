from sklearn import datasets
import random
import numpy as np


def sigmoid(x):

    return 1/(1 + np.exp(-x))


def slp_sgd(tr_X, tr_y, alpha, rep):

    error = []
    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1,100), n)
    w = (np.array(w)-50)/100
    w = w.reshape(tr_X.shape[1], -1)

    for i in range(rep):
        for k in range(tr_X.shape[0]):
            x = tr_X[k,:]
            v = np.matmul(x, w)
            y = sigmoid(v)
            e = tr_y[k, :] - y

            # 가중치 갱신
            for p in range(w.shape[0]):
                for q in range(w.shape[1]):
                    w[p][q] = w[p][q] + alpha * y[q] * (1-y[q]) * e[q] * x[p]


        # print("error", i, np.mean(e))
        error.append(np.mean(e))

    return w, error
from sklearn import datasets
import random
import numpy as np
from sklearn.model_selection import train_test_split

def slp_sgd_mini_batch(tr_X, tr_y, epoch, batch_size, alpha):

    error = []

    n = tr_X.shape[1] * tr_y.shape[1]
    random.seed = 123
    w = random.sample(range(1, 100), n)
    w = (np.array(w) - 50) / 100
    w = w.reshape(tr_X.shape[1], -1)
    delta_w = np.zeros(w.shape)

    mini_batch_size = batch_size

    for i in range(epoch):
        for k in range(tr_X.shape[0]):
            x = tr_X[k, :]
            v = np.matmul(x, w)
            y = sigmoid(v)
            e = tr_y[k, :] - y

            if batch_size != (k+1):


                for p in range(w.shape[0]):
                    for q in range(w.shape[1]):
                        delta_w[p][q] = delta_w[p][q] + alpha * y[q] * (1-y[q]) * e[q] * x[p]

            else:
                delta_w = delta_w / mini_batch_size
                print("{}, {}".format(batch_size, delta_w))
                w = w + delta_w
                batch_size == batch_size + k + 1

        # print("error", i, np.mean(e))
        error.append(np.mean(e))

    return w, error




iris = datasets.load_iris()
X = iris.data
target = iris.target

num = np.unique(target, axis=0)
num = num.shape[0]
y = np.eye(num)[target]
#     print("accuracy : ", np.mean(pred == target))

""" mini_batch_update"""

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
W, error = slp_sgd_mini_batch(X_train, y_train, 50, 10, 0.01)

import matplotlib.pyplot as plt

pred = np.zeros(X_train.shape[0])
result = []
for i in range(X_train.shape[0]):
    v = np.matmul(X_train[i, :], W)
    y = sigmoid(v)

    pred[i] = np.argmax(y)
    result.append(np.argmax(y_train[i]))
print("accuracy : ", np.mean(pred == result))


pred = np.zeros(X_test.shape[0])
result = []
for i in range(X_test.shape[0]):
    v = np.matmul(X_test[i, :], W)
    y = sigmoid(v)

    pred[i] = np.argmax(y)
    result.append(np.argmax(y_test[i]))
print("accuracy : ", np.mean(pred == result))

# W, error = slp_sgd(X, y, alpha=0.01, rep=1000)

# pred = np.zeros(X.shape[0])
# for i in range(X.shape[0]):
#     v = np.matmul(X[i, :], W)
#     y = sigmoid(v)
#
#     pred[i] = np.argmax(y)
#     print("target, predict", target[i], pred[i])
#
# print("accuracy :", np.mean(pred==target))

""" 그래프 그리는 부분"""
# import matplotlib.pyplot as plt
#
# alpha = [0.05, 0.1, 0.5]
# for _alpha in alpha:
#
#     X = iris.data
#     target = iris.target
#
#     num = np.unique(target, axis=0)
#     num = num.shape[0]
#     y = np.eye(num)[target]
#
#     W, error = slp_sgd(X, y, alpha=_alpha, rep=1000)
#     pred = np.zeros(X.shape[0])
#     for i in range(X.shape[0]):
#         v = np.matmul(X[i, :], W)
#         y = sigmoid(v)
#
#         pred[i] = np.argmax(y)
#         # print("target, predict", target[i], pred[i])
#
#
#     plt.plot(error)
#     plt.title("alpha : {0}".format(_alpha))
#     plt.show()
#
