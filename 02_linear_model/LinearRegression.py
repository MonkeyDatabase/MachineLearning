import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_list = [1, 2, 3, 4]
y_list = [2, 4, 6, 8]


def loss(y_predicted, y_actual):
    return (y_predicted - y_actual) * (y_predicted - y_actual)


def forward(x):
    return w * x + b


w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 2.1, 0.1):
        result = 0.0
        for x, y in zip(x_list, y_list):
            y_pre = forward(x)
            result += loss(y_pre, y)
        mse = result / 4
        print('\t', w, b, mse)
        w_list.append(w)
        b_list.append(b)
        mse_list .append(mse)
w_list = np.asarray(w_list)
b_list = np.asarray(b_list)
mse_list = np.asarray(mse_list)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(w_list, b_list, mse_list, c=mse_list)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("mse")
plt.show()



