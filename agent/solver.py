import numpy as np

class LunarLanderSolver(object):
    def __init__(self, env, gamma, alpha, decay, training_rounds):
        self.env = env
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.decay = np.log(decay)
        self.training_rounds = int(training_rounds)

    def solve(self, render=False):
        # TODO
        # Set Up Data Structures
        
        self.epsilon = 0
        for _ in range(self.training_rounds):
            self.play_game(render)
            self.epsilon += self.decay

    def play_game(self, render=False):       
        observation = self.env.reset()
        for _ in range(self.env.spec.timestep_limit):
            observation = self.normalize_observation(observation)
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
        #TODO
        return self.env.action_space.sample()
    
    def add_result(self, observation, action, reward):
        #TODO
        return
        
    def normalize_observation(self, observation):
        #TODO
        return observation
