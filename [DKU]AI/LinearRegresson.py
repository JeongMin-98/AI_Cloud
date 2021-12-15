import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

X = np.array([[1930],[1940],[1950],[1965],[1972],[1982],[1992],[2010],[2016]])
Y = np.array([59,62,70,69,71,74,75,76,78])

reg = linear_model.LinearRegression()

reg.fit(X, Y)

plt.scatter(X, Y, color = 'blue')

y_pred = reg.predict(X)

plt.plot(X, y_pred, color='red', linewidth=3)
plt.show()

print(reg.predict([[1962]]))

a =[]
for i in range(30, 100):
    k = reg.predict([[i]])
    k_ = reg.predict([[i+1]])
    if k*k_ < 0:
        a.append(i)
        break

print(a)