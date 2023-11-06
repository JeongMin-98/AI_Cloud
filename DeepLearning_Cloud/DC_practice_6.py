import numpy as np

def myNN(x):
    w = np.array([0.2, -0.1, 0.3])
    b = 0
    seta = 0

    v = np.sum(x * w) + b
    y = v if v > seta else 0
    return y

ds = np.array([[0.3,0.1,0.8], [0.5,0.6,0.3], [0.1,0.2,0.1], [0.8,0.7,0.7], [0.5,0.5,0.6]])

for i in range(5):
    print("{0}의 output y: {1}".format(i, myNN(ds[i,:])))
