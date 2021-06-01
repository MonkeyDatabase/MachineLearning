import numpy as np
import pandas as pd
from PathEnv import Env as penv


class QLearner(object):
    # Config
    num_states = 0  # count of states
    num_actions = 0  # count of actions
    alpha = 0.2  # learning rate
    gamma = 0.9  # discount rate
    epsilon = 0.9  # greedy policy
    verbose = False  # log mode
    action_names = []  # action names
    max_episode = 13  # max episode
    env: penv = None  # environment

    # Runtime
    q_table = []

    def __init__(self,
                 num_states=10,
                 num_actions=2,
                 alpha=0.2,
                 gamma=0.9,
                 epsilon=0.9,
                 action_names=[],
                 verbose=False,
                 max_episode=1,
                 env=penv()):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.verbose = verbose
        self.max_episode = max_episode
        self.env = env
        if num_actions == len(action_names):
            self.action_names = action_names

    def build_q_table(self):
        q_table = pd.DataFrame(
            np.zeros((self.num_states, self.num_actions)),
            columns=self.action_names if len(self.action_names) > 0 else None
        )
        print(q_table)
        self.q_table = q_table
        return q_table

    def choose_action(self, state):
        state_actions = self.q_table.iloc[state, :]
        if (np.random.uniform() > self.epsilon) or (state_actions.all() == 0):
            print('       greedy')
            action_name = np.random.choice(self.action_names)
        else:
            print('ads')
            action_id = state_actions.argmax()
            action_name = self.action_names[action_id]
        return action_name

    def rl(self):
        self.build_q_table()
        for episode in range(self.max_episode):
            step_counter = 0
            s = 0
            is_terminated = False

            while not is_terminated:
                a = self.choose_action(s)
                s_, r = self.env.get_env_feedback(s, a)
                q_predict = self.q_table.loc[s, a]
                if s_ != 'terminal':
                    q_target = r + self.gamma * self.q_table.iloc[s_, :].max()
                else:
                    q_target = r
                    is_terminated = True

                self.q_table.loc[s, a] += self.alpha * (q_target - q_predict)
                s = s_
                step_counter += 1

                self.env.update_env(s, episode, step_counter)

