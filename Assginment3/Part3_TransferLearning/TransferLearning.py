import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class ProgressiveActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the Progressive Actor-Critic network for three tasks.

        Args:
            input_dim (int): Dimension of the input state.
            output_dim (int): Dimension of the action space.
        """
        super(ProgressiveActorCritic, self).__init__()

        # Pre-trained networks for Task 1 and Task 2 (frozen)
        self.task1_column = self._create_column(input_dim, freeze=True, pretrained_weights=None)
        self.task2_column = self._create_column(input_dim, freeze=True, pretrained_weights=None)

        # New column for Task 3
        self.task3_column = self._create_column(input_dim, freeze=False, pretrained_weights=None)

        # Lateral connections from Task 1 and Task 2 to Task 3
        self.lateral1_to_3 = nn.Linear(128, 128)
        self.lateral2_to_3 = nn.Linear(128, 128)

        # Actor-Critic heads for Task 3
        self.policy_head = nn.Linear(128, output_dim)
        self.value_head = nn.Linear(128, 1)

    def _create_column(self, input_dim, freeze, pretrained_weights):
        """Creates a column for a single task with optional pre-trained weights."""
        column = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        if pretrained_weights:
            # Load pre-trained weights into the column
            state_dict = column.state_dict()
            state_dict.update(pretrained_weights)
            column.load_state_dict(state_dict)

        if freeze:
            for param in column.parameters():
                param.requires_grad = False

        return column

    def lateral_forward(self, x, task1_features, task2_features):
        """
        Computes the lateral contributions to the new task's features.

        Args:
            x (Tensor): Input state.
            task1_features (Tensor): Features from Task 1 column.
            task2_features (Tensor): Features from Task 2 column.

        Returns:
            Tensor: Combined features from lateral connections.
        """
        lateral_features = F.relu(self.lateral1_to_3(task1_features))
        lateral_features += F.relu(self.lateral2_to_3(task2_features))
        return lateral_features

    def forward(self, x):
        """
        Forward pass through the Progressive Actor-Critic network for Task 3.

        Args:
            x (Tensor): Input state.

        Returns:
            dict: Contains the policy (actor) and value (critic) outputs.
        """
        # Features from Task 1 and Task 2 columns
        features_task1 = self.task1_column(x)
        features_task2 = self.task2_column(x)

        # Features from Task 3 column
        features_task3 = self.task3_column(x)

        # Add lateral contributions
        lateral_features = self.lateral_forward(x, features_task1, features_task2)
        features_task3 += lateral_features

        # Actor-Critic heads
        policy = F.softmax(self.policy_head(features_task3), dim=-1)  # Actor output (policy)
        value = self.value_head(features_task3).squeeze(-1)           # Critic output (value)

        return {
            "policy": policy,
            "value": value
        }


def plot_single_reward(episode_rewards, policy_lr, value_lr, discount_factor):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Cumulative Reward per Episode')

    # Compute averaged rewards every 20 episodes
    window_size = 20
    averaged_rewards = [
        np.mean(episode_rewards[i:i + window_size])
        for i in range(0, len(episode_rewards), window_size)
    ]
    averaged_x = [i for i in range(0, len(episode_rewards), window_size)]
    plt.plot(averaged_x, averaged_rewards, marker='o', linestyle='-', color='red', label='Average Reward (Every 20 Episodes)')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Cumulative and Averaged Rewards\nPolicy LR: {policy_lr}, Value LR: {value_lr}, Discount: {discount_factor}')
    plt.legend()
    plt.grid(True)
    plt.show()


# Training loop
def train(env, policy, value_network, discount_factor, max_episodes, max_steps):
    episode_rewards = []
    for episode in range(max_episodes):
        state,_ = env.reset()

        state = pad_with_zeros(state, 6- env.observation_space.shape[0])
        state = torch.tensor(state, dtype=torch.float32)

        cumulative_reward = 0
        for step in range(max_steps):
            action_probs = policy(state)
            action = torch.multinomial(action_probs, 1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Properly handle episode termination

            next_state = pad_with_zeros(next_state, 6 -  env.observation_space.shape[0])
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
                episode_rewards.append(cumulative_reward)
                print(f"Episode {episode} Reward: {cumulative_reward}")

                if episode > 100 and np.mean(episode_rewards[-100:]) > 475:
                    if fine_tunining:
                        torch.save(policy.state_dict(), "Assginment3/Part1_IndividualNet/CartPole_AC/cartpole_policy.pth")
                        torch.save(value_network.state_dict(), "Assginment3/Part1_IndividualNet/CartPole_AC/cartpole_value.pth")
                    return episode_rewards
                break
    return episode_rewards


def main():

    # Example usage:
    input_dim = 10      # Dimension of the state space
    output_dim = 4      # Dimension of the action space

    model = ProgressiveActorCritic(input_dim, output_dim)

    # Example state input
    state = torch.randn(1, input_dim)
    output = model(state)
    print("Policy:", output["policy"])
    print("Value:", output["value"])


    env = gym.make('CartPole-v1')

    policy = PolicyNetwork(state_size=6, action_size=3, learning_rate=0.0001)
    value_network = ValueNetwork(state_size=6, learning_rate=0.0005)
    rewards = train(env, policy, value_network, discount_factor=0.99, max_episodes=1500, max_steps=501)
    plot_single_reward(rewards, policy_lr=0.0001, value_lr=0.0005, discount_factor=0.99)


if __name__ == '__main__':
    main()

