import numpy as np
import matplotlib.pyplot as plt
import random as rand


r = 0
K = 3
T = 3000
R = [1, -10, 0.8]

Q = [0, 0, 0]
count = [0, 0, 0]

e = 0.01
r_list = []
for t in np.arange(0, T):
    k = 0
    if rand.uniform(0, 1) < e:
        k = rand.randint(0, K - 1)
    else:
        for i in np.arange(0, K):
            k = i if Q[i] > Q[k] else k
            print(k)
    v = R[k]
    r += v
    count[k] += 1
    Q[k] = ((count[k] - 1) * Q[k] + v) / count[k]
    r_list.append(r)
plt.plot(r_list, color='grey', label='0.01')

r = 0
e = 0.1
r_list2 = []
for t in np.arange(0, T):
    if rand.uniform(0, 1) < e:
        k = rand.randint(0, K - 1)
    else:
        for i in np.arange(0, K):
            k = i if Q[i] > Q[k] else k
    v = R[k]
    r += v
    count[k] += 1
    Q[k] = ((count[k] - 1) * Q[k] + v) / count[k]
    r_list2.append(r)
plt.plot(r_list2, color='red', label='0.1')

plt.title("e-greedy algorithm\n (threshold greater, exploration more, exploitation less)")
plt.xlabel("times")
plt.ylabel("reward")
plt.legend()
plt.show()
