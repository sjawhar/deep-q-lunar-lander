import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class LunarLanderSolver(object):
    def __init__(self, env, decay, init_episodes, train_episodes, train_rounds):
        self.env = env
        self.decay = np.log(decay)
        self.init_episodes = int(init_episodes)
        self.train_episodes = int(train_episodes)
        self.train_rounds = int(train_rounds)

    def solve(self):
        self.observed = {}
        self.learners = []

        for action in range(self.env.action_space.n):
            self.observed[action] = ([], [])
            self.learners.append(KNeighborsRegressor(weights='distance', n_jobs=-1))
            
        self.epsilon = 0
        self.observe(self.init_episodes, is_decay=False)
        self.train()
        for _ in range(self.train_rounds):
            self.observe(self.train_episodes, is_decay=True)
            print("Done with training set {}".format(_))
    
    def observe(self, episodes, is_decay):
        for _ in range(episodes):
            self.play_game()
            if is_decay == True:
                self.epsilon += self.decay
            if (_ + 1) % 100 == 0:
                print("Done with episode {} {}".format(_ + 1, "of init" if is_decay == False else ""))

    def play_game(self):       
        observation = self.env.reset()
        for _ in range(self.env.spec.timestep_limit):
            prev_observation = observation
            action = self.get_next_action(observation)
            observation, reward, done, info = self.env.step(action)
            self.add_result(prev_observation, action, reward)
            
            if done:
                break
        
    def get_next_action(self, observation):
        if np.log(np.random.random_sample()) < self.epsilon:
            return self.env.action_space.sample()
        
        return self.get_best_action(observation)

    def get_best_action(self, observation):
        observation = observation.reshape(1, -1)
        best_action = 0
        best_reward = self.learners[0].predict(observation)
        for action in range(1, self.env.action_space.n):
            reward = self.learners[action].predict(observation)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action
    
    def add_result(self, observation, action, reward):
        self.observed[action][0].append(observation)
        self.observed[action][1].append(reward)
        return
    
    def train(self):       
        for action in range(self.env.action_space.n):
            X, y = self.observed[action]
            self.learners[action].fit(X, y)

