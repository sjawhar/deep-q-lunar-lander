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
        self.episode_solved = deque([], average)
        self.experience = deque([], experience)

    def solve(self):           
        self.epsilon = 0
        average_reward = 0
        is_train = False
        for episode in range(self.init_episodes + self.train_episodes):
            episode_reward = self.play(is_train)
            self.episode_rewards.append(episode_reward)
            self.episode_solved.append(episode_reward >= 200)
            average_reward = sum(self.episode_rewards)/len(self.episode_rewards)
            print("Episode {} - Reward: {}, Average Reward: {}, Solved: {}".format(
                episode + 1, episode_reward, average_reward, "Yes" if min(solved) else "No"
            ))

            if episode >= self.init_episodes:
                self.epsilon = max(self.min_epsilon, self.epsilon + self.epsilon_decay)
                is_train = True

    def play(self, is_train=False):       
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
        
        possible_rewards = self.learner.predict(observation.reshape(1, -1))
        return possible_rewards.argmax()
    
    def observe(self, previous_observation, action, observation, reward, done):
        experience = np.concatenate((previous_observation, observation, [action, reward, done]), axis=0)
        self.experience.append(experience)
  
    def train(self):
        training_set = np.random.choice(len(self.experience), self.train_size, replace=False)
        
        observation_space = self.env.observation_space.shape[0]
        previous_observations = np.zeros((self.train_size, observation_space))
        observations = np.copy(previous_observations)
        actions = np.zeros((self.train_size, self.env.action_space.n))
        rewards = np.zeros(self.train_size)
        dones = np.zeros(self.train_size)
        
        for i in range(self.train_size):
            experience = training_set[i]
            previous_observations[i,:] = self.experience[experience][:observation_space]
            observations[i,:] = self.experience[experience][observation_space:2*observation_space]
            actions[i, int(self.experience[experience][-3])] = 1
            rewards[i] = self.experience[experience][-2]
            dones[i] = self.experience[experience][-1]

        future_rewards = self.learner.predict(observations).max(axis=1)
        discounted_rewards = self.gamma * (1 - dones) * future_rewards
        rewards += discounted_rewards

        self.learner.fit(previous_observations, actions, rewards)
