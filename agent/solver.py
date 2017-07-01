import numpy as np
from collections import deque

class LunarLanderSolver(object):
    def __init__(self, env, learner, decay, gamma, init_episodes, train_episodes, train_size, average, experience):
        self.env = env
        self.learner = learner

        self.decay = np.log(decay)
        self.gamma = float(gamma)
        self.init_episodes = int(init_episodes)
        self.train_episodes = int(train_episodes)
        self.train_size = int(train_size)
        self.average = float(average)
        self.episode_rewards = deque([], average)
        self.experience = deque([], experience)

    def solve(self, is_record=False):           
        self.epsilon = 0
        self.is_fitted = False
        average_reward = 0
        is_train = False
        for episode in range(self.init_episodes + self.train_episodes):
            episode_reward = self.play(is_train)
            
            if episode + 1 == self.average:
                average_reward = sum(self.episode_rewards)/self.average
            elif episode >= self.average:
                average_reward += (episode_reward - self.episode_rewards.popleft())/self.average
                print("Done with episode {}. Average reward: {}".format(episode + 1, average_reward))
            
            self.episode_rewards.append(episode_reward)

            if episode >= self.init_episodes:
                self.epsilon += self.decay
                is_train = True

    def play(self, is_train):       
        observation = self.env.reset()
        reward, total_reward = 0, 0
        done = False
        while not done:
            if is_train == True:
                self.train()

            action = self.get_next_action(observation, reward, done)
            previous_observation = observation
            observation, reward, done, info = self.env.step(action)
            self.observe(previous_observation, action, observation, reward, done)
            total_reward += reward
        return total_reward
        
    def get_next_action(self, observation, reward, done):
        if np.log(np.random.random_sample()) < self.epsilon:
            return self.env.action_space.sample()
        
        possibilities = np.zeros((self.env.action_space.n, observation.size + 1))
        possibilities[:, :-1] = np.repeat(observation.reshape(1, -1), self.env.action_space.n, axis=0)
        possibilities[:, -1] = range(self.env.action_space.n)
        possible_rewards = self.learner.predict(possibilities)

        return np.argmax(possible_rewards)
    
    def observe(self, previous_observation, action, observation, reward, done):
        self.experience.append((np.append(previous_observation, action), observation, reward, done))
    
    def train(self):
        training_set = np.random.choice(len(self.experience), self.train_size, replace=False)
        obs_act, observations, rewards, dones = [], [], [], []
        for i in training_set:
            experience = self.experience[i]
            obs_act.append(experience[0])
            observations.append(experience[1])
            rewards.append(experience[2])
            dones.append(experience[3])

        if self.is_fitted == True:
            discounted_rewards = [self.gamma * (1 - dones[i]) for i in range(self.train_size)] * self.learner.predict(observations).max(axis = 1)
            rewards += discounted_rewards

        self.learner.fit(obs_act, rewards)
