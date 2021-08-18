import gym
import numpy as np

import dqn_model as dm
import dqn_buffer as db
import dqn_agent as dag
import dqn_algorithm as dal

LEARN_FREQ = 5 # 多少经验训练一次
BUFFER_SIZE = 20000 # buffer_size
MEMORY_WARMUP_SIZE = 200 # 先存储一些经验
BATCH_SIZE = 32
LR_RATE = 1e-3
GAMMA = 0.99 # 奖励因子的衰减值

def run_one_episode(env, algorithm, agent, rpm):
    step = 0
    total_reward = 0
    obs = env.reset()

    while True:
        step += 1
        action = agent.sample(obs)
        s_, r, d, i = env.step(action)
        rpm.append((obs, action, r, s_, d))

        if len(rpm) >= MEMORY_WARMUP_SIZE and step % LEARN_FREQ == 0:
            obs_batch, action_batch, r_batch, s_batch, d_batch = rpm.sample_exp(BATCH_SIZE)
            algorithm.learn(obs_batch, action_batch, r_batch, s_batch, d_batch)

        obs = s_
        total_reward += r
        if d:
            break

    return total_reward

def evaluate(env, agent, render=False):
    step = 0
    total_reward = []

    for i in range(5):

        obs = env.reset()
        episode_reward = 0
        while True:
            if render:
                env.render()
            step += 1
            action = agent.predict(obs)
            s_, r, d, i = env.step(action)
            obs = s_
            episode_reward += r
            if d:
                break
        total_reward.append(episode_reward)
    return np.mean(total_reward)

def main():
    env = gym.make("CartPole-v0")
    action_dim = env.action_space.n # 2
    obs_shape = env.observation_space.shape[0] # 4
    dqn_model = dm.Model(obs_shape, action_dim)
    dqn_alg = dal.DQN(dqn_model, GAMMA, LR_RATE)
    dqn_age = dag.Agent(action_dim, dqn_alg, e_greedy=0.1, e_greedy_dec=1e-6)
    dqn_buffer = db.ReplayBuffer(BUFFER_SIZE)

    while len(dqn_buffer) < MEMORY_WARMUP_SIZE:
        run_one_episode(env, dqn_alg, dqn_age, dqn_buffer)

    max_episode = 2000

    episode = 0

    while episode < max_episode:

        for i in range(50):
            total_reward = run_one_episode(env, dqn_alg, dqn_age, dqn_buffer)
            episode += 1

        eval_reward = evaluate(env, dqn_age, True)

        print("episode: {}   e-greedy: {}   eval_reward: {}". \
              format(episode, dqn_age.e_greedy, eval_reward))

    save_path = "./dqn_model.h5"
    dqn_alg.model.save(save_path)
    env.close()



if __name__ == '__main__':
    main()

