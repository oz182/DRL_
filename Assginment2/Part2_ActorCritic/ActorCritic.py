import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt


# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()

        # Storage for log probabilities and rewards
        self.saved_log_probs = []
        self.rewards = []

        # Storage for state values
        self.saved_values = []

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)  # Softmax across the last dimension


# Select an action based on policy probabilities
def select_action(policy_net, state):
    action_probs = policy_net(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    # Save the log probability of the chosen action
    policy_net.saved_log_probs.append(log_prob)
    return action.item()


# Optimize the policy network based on REINFORCE with baseline algorithm
def optimize_actor_critic(policy_net, value_net, optimizer, value_optimizer, state, action, discount_factor):
    q_value = policy_net(state).gather(1, action)  # Q-values for the policy taken action
    v_value = value_net(state)  # Value of the state
    advantage = q_value - v_value  # Advantage

    # Optimize value network
    value_loss = F.mse_loss(v_value, )
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Policy loss using advantage (G_t - baseline)
    advantage = G_t - state_values.squeeze().detach()
    policy_loss = []
    for log_prob, adv in zip(policy_net.saved_log_probs, advantage):
        policy_loss.append(-log_prob * adv)

    optimizer.zero_grad()
    torch.stack(policy_loss).sum().backward()
    optimizer.step()

    # Clear saved data
    del policy_net.saved_log_probs[:]
    del policy_net.rewards[:]
    del value_net.saved_values[:]


# Plot the rewards over time
def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Advetage Actor-Critic', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Advetage Actor-Critic')
    plt.legend()
    plt.show()


# Main training loop
def main():
    # Hyperparameters
    max_episodes = 1000
    max_steps = 500
    learning_rate = 0.001
    discount_factor = 0.999

    # Initialize environment and policy network
    env = gym.make('CartPole-v1', render_mode=None)
    policy_net = PolicyNet(4, [16, 8], 2)
    optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate)

    # Initialize Q network and optimizer
    Q_net = PolicyNet(4, [16, 8], 2)
    Q_optimizer = optim.AdamW(Q_net.parameters(), lr=learning_rate)

    # Initialize value network and optimizer
    value_net = PolicyNet(4, [16, 8], 1)
    value_optimizer = optim.AdamW(value_net.parameters(), lr=learning_rate)
    
    rewards = []
    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for step in range(max_steps):
            action = select_action(policy_net, state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = torch.tensor(next_state, dtype=torch.float32)

            optimize_actor_critic(policy_net, value_net, optimizer, value_optimizer, state, action, discount_factor)

            if done or truncated:
                break
        
        # Optimize policy after the episode using baseline REINFORCE
        episode_rewards = policy_net.rewards
        rewards.append(sum(episode_rewards))

     
        # Log progress
        print(f"Episode {episode + 1}: Rewards={rewards[-1]}")

    print("Training complete!")
    plot_rewards(rewards)


if __name__ == "__main__":
    main()
