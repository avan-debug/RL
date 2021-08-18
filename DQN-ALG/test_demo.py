import gym
import pytest
import tensorflow as tf
from dqn_algorithm import DQN
import dqn_model as dm
import numpy as np
from logging import getLogger
import dqn_agent as da
import dqn_buffer as db

logger = getLogger()

model_demo = dm.Model(4, 3)
dqn = DQN(model_demo, 0.9, 0.01)
agent = da.Agent(3, dqn)
buffer = db.ReplayBuffer(123)

def test_dqn():
    actions = tf.random.normal([3, 1])
    features = tf.random.normal([3, 4])
    labels = tf.random.normal([3, 1])
    dqn.replace_target()

def test_rand():
    print(np.random.rand())

def test_agent_pred():
    obs = tf.constant([1, 2, 3, 4])
    act = agent.predict(obs)
    print(act)


def test_np():
    a = [1., 2., 3.]
    b = np.array(a).astype("float32")
    print(b)

def test_buffer():
    print(len(buffer))

def test_gym_buffer():
    env = gym.make("CartPole-v0")
    total_step = 0
    for i_episode in range(1):
        env.reset()
        for i in range(100):
            env.render()
            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)
            buffer.append((obs, rew, done, info))
            print(obs, rew, done, info)
            total_step += 1
            if done:
                print("done!")
                break
    env.close()
    print(len(buffer), total_step)
    print(env.observation_space.shape[0])

if __name__ == '__main__':
    test_gym_buffer()
    # pytest.main()