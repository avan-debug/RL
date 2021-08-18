import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=self.max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample_exp(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, actions_batch, rewards_batch, next_obs_batch, done_batch \
            = [], [], [], [], []
        for exp in mini_batch:
            s, a, r, s_, d = exp
            obs_batch.append(s)
            actions_batch.append(a)
            rewards_batch.append(r)
            next_obs_batch.append(s_)
            done_batch.append(d)

        return np.array(obs_batch).astype("float32"), np.array(actions_batch).astype("int32"),\
               np.array(rewards_batch).astype("float32"),np.array(next_obs_batch).astype("float32"), \
               np.array(done_batch).astype("float32")

    def __len__(self):
        return len(self.buffer)
