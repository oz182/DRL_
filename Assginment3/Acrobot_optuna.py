import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import optuna

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
        return F.softmax(x, dim=-1)

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
        return x

def select_action(policy_net, state):
    action_probs = policy_net(state)
    action_dist = Categorical(action_probs)
    action = action_dist.sample()
    log_prob = action_dist.log_prob(action)
    return action.item(), log_prob

def optimize_actor_critic(value_net, optimizer, value_optimizer, LossVector_value, LossVector_policy):
    value_loss = sum(LossVector_value) / len(LossVector_value)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_loss = sum(LossVector_policy) / len(LossVector_policy)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return value_loss, policy_loss

def compute_loss(policy_net, value_net, state, action, reward, next_state, done, log_prob, discount_factor):
    td_target = reward + (1 - done) * discount_factor * value_net(next_state)
    advantage = td_target - value_net(state).detach()

    value_loss = value_net(state) * advantage
    policy_loss = log_prob * advantage.detach()

    return policy_loss, value_loss

def train_model(lr_policy, lr_value, discount_factor):
    env = gym.make('Acrobot-v1', render_mode=None)
    policy_net = PolicyNet(6, [256, 64, 32], 3)
    value_net = ValueNet(6, [256, 64, 32], 1)

    optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)

    max_episodes = 500
    max_steps = 500
    total_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        rewards = 0
        LossVector_value = []
        LossVector_policy = []

        for step in range(max_steps):
            action, log_prob = select_action(policy_net, state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Use potential-based shaping for the reward
            reward = -(np.cos(next_state[0]) + np.cos(next_state[1] + next_state[0]))
            rewards += reward

            next_state = torch.tensor(next_state, dtype=torch.float32)

            policy_loss, value_loss = compute_loss(policy_net, value_net, state, action, reward, next_state, done, log_prob, discount_factor)

            LossVector_policy.append(policy_loss)
            LossVector_value.append(value_loss)

            state = next_state

            if done or truncated:
                break

        total_rewards.append(rewards)
        optimize_actor_critic(value_net, optimizer, value_optimizer, LossVector_value, LossVector_policy)

    return np.mean(total_rewards[-50:])  # Average reward over the last 50 episodes

def objective(trial):
    # Define the search space for hyperparameters
    lr_policy = trial.suggest_loguniform('lr_policy', 1e-5, 1e-2)
    lr_value = trial.suggest_loguniform('lr_value', 1e-5, 1e-2)
    discount_factor = trial.suggest_uniform('discount_factor', 0.9, 0.999)

    # Train the model and return the reward
    average_reward = train_model(lr_policy, lr_value, discount_factor)
    return average_reward

def main():
    # Create the Optuna study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Print the best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best reward:", study.best_value)

    # Visualize the study results
    optuna.visualization.matplotlib.plot_optimization_history(study).show()
    optuna.visualization.matplotlib.plot_param_importances(study).show()

if __name__ == "__main__":
    main()
