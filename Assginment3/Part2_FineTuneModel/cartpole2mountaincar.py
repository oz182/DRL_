import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import optuna
from plotly.io import show
import sklearn
import time

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
        return output[:, :2]

    def get_action(self, state):  # Return action and log(prob(action))
        x = self.forward(state)
        std = torch.exp(x[:, 1])
        dist = torch.distributions.Normal(x[:, 0], std)
        probs = dist.sample()
        action = torch.tanh_(probs)
        return action.detach().numpy(), dist.log_prob(action)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
             nn.init.zeros_(self.fc3.bias)

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

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)


def pad_with_zeros(v, pad_size):
    v = np.asarray(v, dtype=np.float32)  # Ensure v is a NumPy array
    v_t = np.hstack((v, np.zeros(pad_size)))
    return v_t.reshape((1, v_t.shape[0]))

def reinitialize_output_layer(net, output_size, mean=0.0, std=0.1):
    net.fc3 = nn.Linear(net.fc2.out_features, output_size)
    nn.init.normal_(net.fc3.weight, mean=mean, std=std)
    nn.init.normal_(net.fc3.bias, mean=mean, std=std)
    return net

def plot_single_reward(episode_rewards, policy_lr, value_lr, discount_factor):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Cumulative Reward per Episode')

    # Compute averaged rewards every 20 episodes
    window_size = 10
    averaged_rewards = [
        np.mean(episode_rewards[i:i + window_size])
        for i in range(0, len(episode_rewards), window_size)
    ]
    averaged_x = [i for i in range(0, len(episode_rewards), window_size)]
    plt.plot(averaged_x, averaged_rewards, marker='o', linestyle='-', color='red', label='Average Reward (Every 10 Episodes)')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Cumulative and Averaged Rewards\nPolicy LR: {policy_lr}, Value LR: {value_lr}, Discount: {discount_factor}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Training loop
def train(env, policy, value_network, discount_factor, max_episodes, max_steps):

    episode_rewards = []

    episode = 0

    while (episode < max_episodes):

        state,_ = env.reset()

        state = pad_with_zeros(state, 6 -  2)
        state = torch.tensor(state, dtype=torch.float32)

        cumulative_reward = 0

        for step in range(max_steps):
            action,log_prob = policy.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state[1]*=1.005

            next_state = pad_with_zeros(next_state, 6 -  2)
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
            policy_loss = -log_prob * td_error.detach()
            policy.optimizer.zero_grad()
            policy_loss.backward()
            policy.optimizer.step()

            state = next_state
            cumulative_reward += reward

            if done or step == 999:
                episode_rewards.append(cumulative_reward)
                print(f"Episode {episode} Reward: {cumulative_reward}")

                #if episode > 100 and np.mean(episode_rewards[-100:]) > 60:

                    #return episode_rewards
                break

        if episode_rewards!=[] and ((max(episode_rewards) < 1) and (episode > 3)):
            episode = 0
            policy.reset_parameters()
            value_network.reset_parameters()
            print("Resetting the weights")
            episode_rewards = []

        episode += 1

    return episode_rewards


def main():
    #np.random.seed(23)
    #torch.manual_seed(23)


    env = gym.make('MountainCarContinuous-v0', render_mode=None)
    path_actor = "C:/Users/idogu/PycharmProjects/PythonProject/weights/cartpole_policy.pth"
    policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=0.00001)
    policy.load_state_dict(torch.load(path_actor, weights_only=True))
    policy = reinitialize_output_layer(policy, output_size=3, mean=0.0, std=0.1)

    path_critic = "C:/Users/idogu/PycharmProjects/PythonProject/weights/cartpole_value.pth"
    value_network = ValueNetwork(state_size=6, learning_rate=0.0005)
    value_network.load_state_dict(torch.load(path_critic, weights_only=True))
    value_network = reinitialize_output_layer(value_network, output_size=1, mean=0.0, std=0.1)

    start_time = time.time()
    rewards = train(env, policy, value_network, discount_factor=0.99, max_episodes=400, max_steps=999)
    end_time = time.time()
    print("Training time:", end_time - start_time)
    plot_single_reward(rewards, policy_lr=0.00001, value_lr=0.0005, discount_factor=0.99)


if __name__ == '__main__':
    main()