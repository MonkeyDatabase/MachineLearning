import numpy as np
import matplotlib.pyplot as plt
import math
import random


def reward_1():
    if random.uniform(0, 1) < 0.6:
        return 0
    else:
        return 1


def reward_2():
    if random.uniform(0, 1) < 0.2:
        return 1
    else:
        return 0


# def reward_3():
#     if random.uniform(0, 1) < 0.4:
#         return 8
#     else:
#         return -3


#R = [reward_1, reward_2, reward_3]              # Reward Functions
R = [reward_1, reward_2]              # Reward Functions


def soft_max(q_list, tau):
    p_list = []
    _sum = 0
    for i in np.arange(len(q_list)):
        _sum += math.exp(q_list[i] / tau)
    for i in np.arange(len(q_list)):
        p_temp = math.exp(q_list[i] / tau) / _sum
        p_list.append(p_temp)
    np.set_printoptions(suppress=True)
    print('q_list:', q_list, '\n  -> p_list: ', p_list)
    return p_list


def choose(p_list):
    p_temp = random.uniform(0, 1)
    for i in np.arange(len(p_list)):
        p_temp -= p_list[i]
        if p_temp < 0:
            return i


def append_r_list(r_list, r_temp):
    r_last = r_list[len(r_list) - 1] if len(r_list) != 0 else 0
    r_latest = r_last + r_temp
    r_list.append(r_latest)


def refresh_q_list(q_list, count_list, k, r_temp):
    q_before = q_list[k]
    count_after = count_list[k]
    q_after = (q_before * (count_after - 1) + r_temp) / count_after
    q_list[k] = q_after


def rl_soft_max(t_total, choices_rewards, tau):
    q_list = [0] * len(choices_rewards)
    r_list = []
    count_list = [0] * len(choices_rewards)
    for t in np.arange(t_total):
        p_list = soft_max(q_list, tau)
        k = choose(p_list)
        print('chosed :', k)
        count_list[k] += 1
        r_temp = choices_rewards[k]()
        append_r_list(r_list, r_temp)
        refresh_q_list(q_list, count_list, k, r_temp)
    return r_list, count_list


def draw_list(data_list):
    plt.plot(data_list)
    plt.show()


def draw_lists(data_lists, label_list):
    for serial in np.arange(len(data_lists)):
        plt.plot(data_lists[serial], label=label_list[serial])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    r_lists = []
    title_list = []
    for i in np.arange(0.01, 1.01, 0.1):
        r_list_r, count_list_r = rl_soft_max(3000, R, i)
        r_lists.append(r_list_r)
        title_list.append('tau is {:.2f}, count_list is {}'.format(i, count_list_r))
    draw_lists(r_lists, title_list)
