import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='rgb_array')
env.reset()
env.render()
state_dim = env.observation_space.shape[0]  # 4-dimensional state
action_dim = env.action_space.n  # 2 possible actions