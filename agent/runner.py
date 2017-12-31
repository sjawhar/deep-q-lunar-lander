def main():
    from solver import LunarLanderSolver
    from learner import MultiOutNN
    import gym
    env = gym.make('LunarLander-v2')
    
    layer_sizes = (
        env.observation_space.shape[0],
        25 * env.observation_space.shape[0],
        25 * env.observation_space.shape[0],
        env.action_space.n
    )
    learning_rate_init= 0.001
    weight_init = 0.1
    bias_init = -0.001

    learner = MultiOutNN(
        layer_sizes = layer_sizes,
        learning_rate_init = learning_rate_init,
        weight_init = weight_init,
        bias_init = bias_init
    )

    init_episodes = 100
    train_episodes = 2500
    epsilon_decay = 0.99
    min_epsilon = 0.1
    gamma = 0.99
    train_size = 128
    average = 100
    experience = 50000

    solver = LunarLanderSolver(
        env = env,
        learner = learner,
        epsilon_decay = epsilon_decay,
        min_epsilon = min_epsilon,
        gamma = gamma,
        init_episodes = init_episodes,
        train_episodes = train_episodes,
        train_size = train_size,
        average = average,
        experience = experience
    )
    solver.solve()
    
if __name__ == "__main__":
    main()
