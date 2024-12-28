import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

import os

LEARNING_RATE_POLICY = 0.0009
LEARNING_RATE_VALUE = 0.0009
DISCOUNT_FACTOR = 0.999

# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=-1)  # Softmax across the last dimension
    

# Define the value network
class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x  # Softmax across the last dimension


# Select an action based on policy probabilities
def select_action(policy_net, state):
    action_probs = policy_net(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    return action.item(), log_prob


# Optimize the policy network based on A2C algorithm
def optimize_actor_critic(value_net, optimizer, value_optimizer, LossVector_value, LossVector_policy):

    # Optimize value network
    value_loss = sum(LossVector_value) / len(LossVector_value)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    # Policy loss using advantage 
    policy_loss = sum(LossVector_policy) / len(LossVector_policy)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return value_loss, policy_loss


def compute_loss(policy_net, value_net, state, action, reward, next_state, done, truncated, ImportanceSampling, log_prob):
    action_tensor = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)

    td_target = reward + (1 - done) * DISCOUNT_FACTOR * value_net(next_state) # TD Target
    advantage = td_target - value_net(state).detach() # Advantage

    # Optimize value network
    value_loss = value_net(state) * advantage #* ImportanceSampling

    # Policy loss using advantage 
    policy_loss = log_prob * advantage.detach() #* ImportanceSampling

    return [policy_loss, value_loss]


# Plot the rewards over episodes
def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Advetage Actor-Critic', alpha=0.7)
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.title('Advetage Actor-Critic')
    plt.legend()
    plt.show()

# Plot the losses over epsidoes
def plot_losses(policy_loss, value_loss):
    plt.figure(figsize=(12, 6))
    plt.plot(policy_loss, label='Policy Loss', alpha=0.7)
    plt.plot(value_loss, label='Value Loss', alpha=0.7)
    plt.xticks([])
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Advetage Actor-Critic')
    plt.legend()
    plt.show()


# Main training loop
def main():
    # Hyperparameters
    max_episodes = 1000
    max_steps = 500

    # Initialzie environment
    env = gym.make('CartPole-v1', render_mode=None)

    # Initialize policy network and optimizer
    policy_net = PolicyNet(4, [256, 128, 32], 2)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE_POLICY)
    policy_loss_all = []

    # Initialize value network and optimizer
    value_net = ValueNet(4, [256, 128, 32], 1)
    value_optimizer = optim.AdamW(value_net.parameters(), lr=LEARNING_RATE_VALUE)
    value_loss_all = []
    
    rewards = 0
    episode_rewards = []

    for episode in range(max_episodes):
        
        ImportanceSampling = 1

        LossVector_value = []
        LossVector_policy = []

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for step in range(max_steps):
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, truncated, _ = env.step(action)

            state = torch.tensor(next_state, dtype=torch.float32)
            rewards += reward
            ImportanceSampling *= DISCOUNT_FACTOR

            LossVector_policy.append(compute_loss(policy_net, value_net, state, action, reward, next_state, done, truncated, ImportanceSampling, log_prob)[0])
            LossVector_value.append(compute_loss(policy_net, value_net, state, action, reward, next_state, done, truncated, ImportanceSampling, log_prob)[1])

            if done or truncated or step == max_steps - 1:
                episode_rewards.append(rewards)
                rewards = 0
                break
        
        value_loss_eps, policy_loss_eps = optimize_actor_critic(value_net, optimizer, value_optimizer, LossVector_value, LossVector_policy)

        value_loss_all.append(float(value_loss_eps))
        policy_loss_all.append(float(policy_loss_eps))

        # Average the losses over the last 20 episodes
        avg_steps = 20
        policy_loss_avg = [np.mean(policy_loss_all[i:i+avg_steps]) for i in range(0, len(policy_loss_all), avg_steps)]
        value_loss_avg = [np.mean(value_loss_all[i:i+avg_steps]) for i in range(0, len(value_loss_all), avg_steps)]

        # Log progress
        print(f"Episode {episode + 1}: Rewards={episode_rewards[-1]}")

    print("Training complete!")
    os.system('echo -e "\a"')  # Terminal beep
    plot_rewards(episode_rewards)
    plot_losses(policy_loss_avg, value_loss_avg)


if __name__ == "__main__":
    main()
