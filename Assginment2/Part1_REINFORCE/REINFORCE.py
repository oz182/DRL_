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


# Optimize the policy network
def optimize_net(policy_net, optimizer, discount_factor):
    R = 0
    G_t = deque()

    # Compute cumulative discounted rewards in reverse
    for r in policy_net.rewards[::-1]:
        R = r + discount_factor * R
        G_t.appendleft(R)

    # Convert to a tensor and normalize
    G_t = torch.tensor(G_t, dtype=torch.float32)
    #G_t = (G_t - G_t.mean()) / (G_t.std() + 1e-5) if len(G_t) > 1 else G_t
    
    # Compute policy loss
    policy_loss = []
    for log_prob, reward in zip(policy_net.saved_log_probs, G_t):
        policy_loss.append(-log_prob * reward)

    # Perform backpropagation and optimization
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    # Clear saved log probabilities and rewards
    del policy_net.saved_log_probs[:]
    del policy_net.rewards[:]


# Optimize the policy network based on REINFORCE with baseline algorithm
def optimize_with_baseline(policy_net, value_net, optimizer, value_optimizer, discount_factor):
    R = 0
    G_t = deque()
    for r in policy_net.rewards[::-1]:
        R = r + discount_factor * R
        G_t.appendleft(R)

    G_t = torch.tensor(G_t, dtype=torch.float32)

    # Optimize value network (baseline)
    state_values = torch.stack(value_net.saved_values).squeeze()
    value_loss = F.mse_loss(state_values.squeeze(), G_t)
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
    plt.plot(rewards, label='REINFORCE', alpha=0.7)
    #plt.plot(rewards, label='REINFORCE with baseline', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    #plt.title('REINFORCE')
    plt.title('REINFORCE with Baseline')
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

    # Initialize value net for baseline
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

            # Save value of state (only with baseline)
            value_net.saved_values.append(value_net(state))

            # Save reward and move to the next state
            policy_net.rewards.append(reward)
            state = torch.tensor(next_state, dtype=torch.float32)

            if done or truncated:
                break

        # Optimize policy after the episode
        episode_rewards = policy_net.rewards
        rewards.append(sum(episode_rewards))
        optimize_net(policy_net, optimizer, discount_factor)
        

        # Optimize policy after the episode using baseline REINFORCE
        """
        episode_rewards = policy_net.rewards
        rewards.append(sum(episode_rewards))
        optimize_with_baseline(policy_net, value_net, optimizer, value_optimizer, discount_factor)
        """

        # Log progress
        print(f"Episode {episode + 1}: Rewards={rewards[-1]}")

    print("Training complete!")
    plot_rewards(rewards)


if __name__ == "__main__":
    main()
