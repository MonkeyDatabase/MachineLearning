import matplotlib.pyplot as plt
import random

x_list = [1, 2, 3, 4]
y_list = [3, 5, 7, 9]
n = 4

w = random.randint(-100000, 100000)
b = random.randint(-100000, 100000)
alpha = 0.01
limit = 0.01


# Linear Regression
def forward(x):
    return w * x + b


def loss(x, y):
    return (forward(x) - y) * (forward(x) - y)


def mse():
    result = 0
    for x, y in zip(x_list, y_list):
        result += loss(x, y)
    return result / n


def gradient_w():
    result = 0
    for x, y in zip(x_list, y_list):
        result += (2 * (forward(x) - y) * x)
    return result / n


def gradient_b():
    result = 0
    for x, y in zip(x_list, y_list):
        result += (2 * (forward(x) - y))
    return result / n


def gradient_descent():
    global w
    global b
    w = w - alpha * gradient_w()
    b = b - alpha * gradient_b()
    return


epoch = 0
epoch_list = []
mse_list = []
while abs(gradient_b()) > limit and abs(gradient_w()) > limit:
    epoch += 1
    epoch_list.append(epoch)
    mse_list.append(mse())
    gradient_descent()
    print('w:', w, ' b:', b, ' mse:', mse())

fig = plt.figure()
plt.plot(epoch_list, mse_list)
plt.show()

