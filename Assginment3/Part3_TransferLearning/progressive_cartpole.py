
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
from plotly.io import show
import sklearn
import time

class PolicyNetwork(nn.Module):
    """
    Simple policy network for both Acrobot and MountainCarContinuous.
    """
    def __init__(self, state_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class ProgressivePolicyNetwork(nn.Module):
    """
    Progressive Neural Network with lateral connections from two source tasks to the target task.
    """
    def __init__(self, state_size, action_size, learning_rate):
        super(ProgressivePolicyNetwork, self).__init__()

        # Source models with different hidden layer sizes
        self.acrobot_model = self.load_pretrained_model('C:/Users/idogu/PycharmProjects/PythonProject/weights/acrobot_policy.pth',
                                                        state_size, hidden_size=12, action_size=action_size)

        self.mountaincar_model = self.load_pretrained_model('C:/Users/idogu/PycharmProjects/PythonProject/weights/mountcar_policy.pth',
                                                            state_size, hidden_size=16, action_size=action_size)

        # Target (CartPole) model with 16 neurons in hidden layers
        self.fc1_target = nn.Linear(state_size, 16)
        self.fc2_target = nn.Linear(16, 16)
        self.fc3_target = nn.Linear(16, action_size)

        # Lateral connections to map source outputs to the target network
        self.L12_lateral_acro2cart = nn.Linear(12, 16)       # From Acrobot (16) → Target (16)
        self.L12_lateral_mountaincar2cart = nn.Linear(16, 16)
        self.L12_lateral_mountaincar2cart = nn.Linear(16, action_size)
        # From MountainCar (12) → Target (16)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def load_pretrained_model(self, filepath, state_size, hidden_size, action_size):
        """
        Load the pretrained model and freeze its weights.
        """
        model = PolicyNetwork(state_size, hidden_size, action_size)
        model.load_state_dict(torch.load(filepath))

        # Freeze the source model
        for param in model.parameters():
            param.requires_grad = False

        return model

    def forward(self, state):
        # Pass through the frozen Acrobot model
        acrobot_output = torch.relu(self.acrobot_model.fc1(state))
        acrobot_output = torch.relu(self.acrobot_model.fc2(acrobot_output))

        # Pass through the frozen MountainCar model
        mountaincar_output = torch.relu(self.mountaincar_model.fc1(state))
        mountaincar_output = torch.relu(self.mountaincar_model.fc2(mountaincar_output))

        # Pass through the target network with lateral connections
        h1_target = torch.relu(
            self.fc1_target(state) +
            self.lateral_acrobot(acrobot_output) +
            self.lateral_mountaincar(mountaincar_output)
        )

        h2_target = torch.relu(self.fc2_target(h1_target))
        output = self.fc3_target(h2_target)

        return torch.softmax(output, dim=-1)