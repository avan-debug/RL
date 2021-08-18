import numpy as np
import tensorflow as tf

class Agent:
    def __init__(self, action_dim, algorithm, e_greedy=0.1, e_greedy_dec=0):
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.e_greedy = e_greedy
        self.e_greedy_dec = e_greedy_dec

    def sample(self, obs):
        sample_rand = np.random.rand()
        if sample_rand < self.e_greedy:
            action = np.random.randint(self.action_dim)
        else:
            action = self.predict(obs)

        # 探索程度逐渐降低
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_dec)

        return action

    def predict(self, obs):
        obs = tf.expand_dims(obs, axis=0)
        pred_val = self.algorithm.model(obs)
        return np.argmax(pred_val)
