import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import collections


# Policy Network (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, action_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return torch.softmax(output[:, :2], dim=1)

# Value Network (Critic)
class ValueNetwork(nn.Module):
    def __init__(self, state_size, learning_rate):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

# Pad state with zeros
def pad_with_zeros(v, pad_size):
    v = np.asarray(v, dtype=np.float32)  # Ensure v is a NumPy array
    v_t = np.hstack((v, np.zeros(pad_size)))
    return v_t.reshape((1, v_t.shape[0]))

# Training loop
def train(env, policy, value_network, discount_factor, max_episodes, max_steps, render, save_model):
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    episode_rewards = np.zeros(max_episodes)

    for episode in range(max_episodes):
        state,_ = env.reset()

        state = pad_with_zeros(state, 2)
        state = torch.tensor(state, dtype=torch.float32)

        cumulative_reward = 0
        for step in range(max_steps):
            action_probs = policy(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination

            next_state = pad_with_zeros(next_state, 6 - 4)
            next_state = torch.tensor(next_state, dtype=torch.float32)

            current_value = value_network(state)
            next_value = value_network(next_state)
            td_target = reward + (1 - done) * discount_factor * next_value
            td_error = td_target - current_value

            # Update Value Network
            value_loss = nn.functional.mse_loss(current_value, td_target.detach())
            value_network.optimizer.zero_grad()
            value_loss.backward()
            value_network.optimizer.step()

            # Update Policy Network
            log_prob = torch.log(action_probs.squeeze(0)[action])
            policy_loss = -log_prob * td_error.detach()
            policy.optimizer.zero_grad()
            policy_loss.backward()
            policy.optimizer.step()

            state = next_state
            cumulative_reward += reward

            if done:
                episode_rewards[episode] = cumulative_reward
                print(f"Episode {episode} Reward: {cumulative_reward}")
                if save_model and np.mean(episode_rewards[-100:]) > 475:
                    torch.save(policy.state_dict(), "./weights/cartpole_policy.pth")
                    torch.save(value_network.state_dict(), "./weights/cartpole_value.pth")
                    solved = True
                break


        if solved:
            break

if __name__ == '__main__':
    np.random.seed(23)
    torch.manual_seed(23)


    env = gym.make('CartPole-v1')
    policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=0.0001)
    value_network = ValueNetwork(state_size=6, learning_rate=0.0005)

    train(env, policy, value_network, discount_factor=0.99, max_episodes=1500, max_steps=501, render=True, save_model=True)
