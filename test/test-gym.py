import gym
env = gym.make('CartPole-v0')
for i_episode in range(5):
    print("episode "+str(i_episode))
    observation = env.reset()
    for t in range(100):
        env.render()
        print("observation: "+str(observation))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("action: "+str(action)+" reward: "+str(reward)+" done: "+str(done)+" info: "+str(info))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

if __name__ == '__main__':
    pass