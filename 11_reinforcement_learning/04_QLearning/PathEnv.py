import time


class Env(object):
    # Config
    num_states = 0  # count of states
    num_actions = 0  # count of actions
    fresh_time = 0  # time slot between actions

    def __init__(self,
                 num_states=0,
                 num_actions=0,
                 fresh_time=0
                 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.fresh_time = fresh_time

    def get_env_feedback(self, s, a):
        if a == 'right':
            if s == self.num_states - 2:
                s_ = 'terminal'
                r = 1
            else:
                s_ = s + 1
                r = -1
        else:
            r = -1
            if s == 0:
                s_ = s
            else:
                s_ = s - 1
        return s_, r

    def update_env(self, s, episode, step_counter):
        env_list = ['-'] * (self.num_states - 1) + ['T']
        if s == 'terminal':
            interaction = 'Episode %s, total_steps = %s' %(episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                          ', end='')
        else:
            env_list[s] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(self.fresh_time)
