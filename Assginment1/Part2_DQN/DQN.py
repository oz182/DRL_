import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

env = gym.make("CartPole-v1", render_mode='human')
#env.reset()
#env.render()
state_dim = env.observation_space.shape[0]  # 4-dimensional state
action_dim = env.action_space.n  # 2 possible actions


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)  # Output Q-values for all actions
    

class ReplayBuffer:
    # Class for creating and hendling the expiriance buffer for the
    # network training. Deque object is used to replace old expiriance with new ones
    # while maintaining a fixed size.

    # Initialzie buffer with a maximum size
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, transition):
        self.buffer.append(transition)

    # Get a sample (of a defined size) from the que
    def sample(self, batch_size):

        # Sample a batch of tarnsition from the buffer
        batch = random.sample(self.buffer, batch_size)

        # "zip(*batch)" is used to zip together all the arrtibutes of one sample from the 
        # batch (the '*' sign is used to unpack the tuples of 1 sample in to componants)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Each componant is then converted into a pytorch tensor for handling with torch module
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)
    


