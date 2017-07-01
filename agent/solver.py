import numpy as np
from collections import deque

class LunarLanderSolver(object):
    def __init__(
            self,
            env,
            learner,
            epsilon_decay,
            min_epsilon,
            gamma,
            init_episodes,
            train_episodes,
            train_size,
            average,
            experience
    ):
        self.env = env
        self.learner = learner

        self.epsilon_decay = np.log(epsilon_decay)
        self.min_epsilon = np.log(min_epsilon)
        self.gamma = float(gamma)
        self.init_episodes = int(init_episodes)
        self.train_episodes = int(train_episodes)
        self.train_size = int(train_size)
        self.episode_rewards = deque([], average)
        self.experience = deque([], experience)

    def solve(self):           
        self.epsilon = 0
        self.is_fitted = False
        average_reward = 0
        is_train = False
        for episode in range(self.init_episodes + self.train_episodes):
            episode_reward = self.play(is_train)
            self.episode_rewards.append(episode_reward)
            average_reward = sum(self.episode_rewards)/len(self.episode_rewards)
            print("Episode {}: Reward: {}, Average reward: {}".format(episode + 1, episode_reward, average_reward))            

            if episode >= self.init_episodes:
                self.epsilon = max(self.min_epsilon, self.epsilon + self.epsilon_decay)
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
        
        possibilities = self.get_possibilities(observation.reshape(1, -1))
        possible_rewards = self.learner.predict(possibilities)

        return np.argmax(possible_rewards)
    
    def get_possibilities(self, observations):
        possibilities = np.zeros((self.env.action_space.n * observations.shape[0], observations.shape[1] + 1))
        possibilities[:, :-1] = np.repeat(observations, self.env.action_space.n, axis=0)
        possibilities[:, -1] = np.tile(range(self.env.action_space.n), observations.shape[0])
        return possibilities
    
    def observe(self, previous_observation, action, observation, reward, done):
        obs_act = np.append(previous_observation, [action])
        self.experience.append((obs_act, observation, reward, done))
  
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
            observations = np.array(observations)
            possibilties = self.get_possibilities(observations)
            future_rewards = self.learner.predict(possibilties).reshape(observations.shape[0], 4).max(axis=1)
            discounted_rewards = self.gamma * (1 - np.array(dones)) * future_rewards
            rewards += discounted_rewards

        self.learner.partial_fit(obs_act, rewards)
        self.is_fitted = True
