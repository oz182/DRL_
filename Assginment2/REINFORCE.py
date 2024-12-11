import random
import numpy as np
import gymnasium as gym
from collections import defaultdict, namedtuple, deque

from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class policy_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(policy_net, self).__init__()
        # Define layers
        self.L1 = nn.Linear(input_size, hidden_size[0])

        self.L2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.L3 = nn.Linear(hidden_size[1], output_size)
        self.activation = nn.ReLU()

        self.episode_rewards = []
        self.saved_logs_probs = []

    def forward(self, x):
        x = self.activation(self.L1(x))  # Hidden layer 1
        x = self.activation(self.L2(x))  # Hidden layer 2
        x = self.L3(x)  # Hidden layer 3
        return F.softmax(x)


def select_action(policy_net, state):
    action_prob = policy_net(state)  # Ganerate an action "score" from state
    action_dist = Categorical(action_prob)  # convert action to categorical propabilties
    action = action_dist.sample()  # sample an action accordingly
    log_prob = action_dist.log_prob(action)  # Select an action according to to propabilty and add log

    policy_net.saved_logs_probs.append(log_prob)

    return action


def OptimizeNet(policy_net, optimizer):

    optimizer.zero_grad()

    R = 0
    G_t = []

    for r in policy_net.episode_rewards[::-1]:
        R =  r + R * DISCOUNT_FACTOR
        G_t.appendleft(R)

    #loss = .log_prob(action) * reward
    #for 

    #loss = 
    G_t.backward()

    optimizer.step()



    pass




MAX_EPISODS = 100
MAX_STEPS = 100
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE)


env = gym.make('CartPole-v1', render_mode='rgb_array')

def main():

    for i_episodes in range(MAX_EPISODS):

        state, _ = env.reset()
        del policy_net.episode_rewards[:]
        del policy_net.saved_logs_probs[:]

        for step in range(MAX_STEPS):
            state = torch.tensor(state, dtype=torch.float32, device="cpu")
            state, reward, done, _ = env.step(select_action(policy_net, state))

            policy_net.episode_rewards.append(reward)

            if done:
                break

        OptimizeNet(policy_net)



if __name__ == '__main__':
    main()