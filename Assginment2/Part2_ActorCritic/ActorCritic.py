import numpy as np
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.999

# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)  # Softmax across the last dimension
    

# Define the policy network
class QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x  # Softmax across the last dimension
    

# Define the policy network
class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x  # Softmax across the last dimension


# Select an action based on policy probabilities
def select_action(policy_net, state):
    action_probs = policy_net(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)

    return action.item(), log_prob


def train_Qnet(Q_net, Q_optimizer, state, next_state, action, reward, done):
    action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)
    q_values = Q_net(state).gather(0, action_tensor)  # Q-values for taken actions
    next_q_values = Q_net(torch.tensor(next_state)).max(0).values.detach()  # Detach target values

    targets = reward + DISCOUNT_FACTOR * next_q_values * (1 - done)

    loss = F.mse_loss(q_values, targets)

    Q_optimizer.zero_grad()
    loss.backward()
    Q_optimizer.step()
    return loss.item()


# Optimize the policy network based on A2C algorithm
def optimize_actor_critic(value_net, Q_net, optimizer, value_optimizer, state, action, log_prob):

    action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0)
    q_value = Q_net(state).gather(0, action_tensor)  # Q-values for the policy taken action

    v_value = value_net(state)  # Value of the state
    advantage = q_value - v_value  # Advantage

    # Optimize value network
    value_loss = -(v_value * advantage * DISCOUNT_FACTOR * LEARNING_RATE)
    value_optimizer.zero_grad()
    value_loss.backward(retain_graph=True)
    value_optimizer.step()

    # Policy loss using advantage 
    policy_loss = -log_prob * advantage.detach()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


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

    # Initialzie environment
    env = gym.make('CartPole-v1', render_mode=None)

    # Initialize policy network and optimizer
    policy_net = PolicyNet(4, [16, 8], 2)
    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE)

    # Initialize Q network and optimizer
    Q_net = QNet(4, [16, 8], 2)
    Q_optimizer = optim.AdamW(Q_net.parameters(), lr=LEARNING_RATE)

    # Initialize value network and optimizer
    value_net = ValueNet(4, [16, 8], 1)
    value_optimizer = optim.AdamW(value_net.parameters(), lr=LEARNING_RATE)
    
    rewards = 0
    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        for step in range(max_steps):
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            optimize_actor_critic(value_net, Q_net, optimizer, value_optimizer, state, action, log_prob)
            train_Qnet(Q_net, Q_optimizer, state, next_state, action, reward, done)

            state = torch.tensor(next_state, dtype=torch.float32)

            rewards += reward

            if done or truncated or step == max_steps - 1:
                episode_rewards.append(rewards)
                rewards = 0
                break
     
        # Log progress
        print(f"Episode {episode + 1}: Rewards={episode_rewards[-1]}")

    print("Training complete!")
    plot_rewards(rewards)


if __name__ == "__main__":
    main()
