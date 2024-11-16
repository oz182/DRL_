import gymnasium as gym

# Create a new game environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
observation, info = env.reset()

#env.close()
print("Done")


