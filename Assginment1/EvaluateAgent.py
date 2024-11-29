import torch
from matplotlib import pyplot as plt

def evaluate_agent(env, policy_net, num_episodes=100):
    """
    Evaluates the agent over a specified number of episodes.
    Args:
        env: The environment to evaluate the agent on.
        policy_net: The trained policy network.
        num_episodes: Number of episodes to test the agent.
    Returns:
        A list of rewards obtained in each episode.
    """
    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = policy_net(state).argmax(dim=1).item()
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward
        
        rewards.append(total_reward)
    
    return rewards

def plot_test_rewards(rewards):
    """
    Plots rewards obtained during the evaluation phase.
    Args:
        rewards: A list of rewards obtained in each episode.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Reward per episode', marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation: Reward per Episode')
    plt.grid()
    plt.legend()
    plt.show()