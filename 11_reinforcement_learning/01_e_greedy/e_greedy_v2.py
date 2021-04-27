import numpy as np
import matplotlib.pyplot as plt
import random


def reward_1():
    if random.uniform(0, 1) < 0.5:
        return 5
    else:
        return -6


def reward_2():
    if random.uniform(0, 1) < 0.3:
        return 10
    else:
        return -5


def reward_3():
    if random.uniform(0, 1) < 0.4:
        return 8
    else:
        return -3


R = [reward_1, reward_2, reward_3]              # Reward Functions
T = 3000                                        # TRY Times
thresholds = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]    # threshold []
reward_lists = []                               # totalReward list<list<int>>
count_lists = []                                # choiceCount list<list<int>>
Q_lists = []                                    # Reward Rate list<list<int>>


def setup_env():
    for i in np.arange(len(thresholds)):
        Q_lists.append([0] * len(R))
        count_lists.append([0] * len(R))
        reward_lists.append([0])


def get_latest_reward(threshold_serial):
    # print('[get_latest_reward]', reward_lists[threshold_serial])
    return reward_lists[threshold_serial][len(reward_lists[threshold_serial]) - 1]


def append_latest_reward(threshold_serial, reward_temp):
    # print(get_latest_reward(threshold_serial))
    reward_lists[threshold_serial].append(get_latest_reward(threshold_serial) + reward_temp)


def el():
    for threshold_serial in np.arange(len(thresholds)):
        k = 0
        if random.uniform(0, 1) < thresholds[threshold_serial]:
            k = random.randint(0, len(R) - 1)
        else:
            for i in np.arange(len(R)):
                k = i if Q_lists[threshold_serial][i] > Q_lists[threshold_serial][k] else k
        reward_temp = R[k]()
        count_lists[threshold_serial][k] += 1
        Q_lists[threshold_serial][k] = (Q_lists[threshold_serial][k] * (count_lists[threshold_serial][k] - 1) + reward_temp) / count_lists[threshold_serial][k]
        append_latest_reward(threshold_serial, reward_temp)


def draw():
    for i in np.arange(len(thresholds)):
        print('[plt] draw threshold, i =', i, 'reward_list:', reward_lists[i])
        plt.plot(reward_lists[i], label=thresholds[i])
    plt.title("e-greedy algorithm\n (threshold greater, exploration more, exploitation less)")
    plt.xlabel("times")
    plt.ylabel("reward")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    setup_env()
    for t_serial in np.arange(T):
        el()
        print(Q_lists)
    draw()

