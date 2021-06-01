import numpy as np
import pandas as pd
import time
import QLearner as ql
from PathEnv import Env as penv

# CONFIG
np.random.seed(2)  # random seed of Pseudorandom

# Globals
N_STATE = 6  # the states of an agent
ACTIONS = ['left', 'right']  # the actions that an agent can do in the environment
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODE = 13  # max episode
FRESH_TIME = 0.3  # slot between two action


if __name__ == '__main__':
    env = penv(num_states=N_STATE, num_actions=len(ACTIONS), fresh_time=FRESH_TIME)
    q = ql.QLearner(num_states=N_STATE, num_actions=len(ACTIONS), alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                    action_names=ACTIONS, max_episode=MAX_EPISODE, env=env)
    q.rl()

