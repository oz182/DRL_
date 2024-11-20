import numpy as np
import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt

class FrozenAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action, 
        based on the Q-Learning algorithm written 
        in the assginment: "Figure 3 - Q-learning algorithm """

        if terminated == True:
            target_function = reward

        else:
            target_function = reward + self.discount_factor * np.max(self.q_values[next_obs])

        self.q_values[obs][action] = (1 - self.lr) * self.q_values[obs][action] + self.lr * target_function

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Main function for playing the game
def run_game(agent):

    # Initialize lists
    eps_rewards_list = []
    steps_list = []

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        steps_counter = 0
        total_rewards_eps = 0

        # play one episode
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_rewards_eps += reward

            # update the agent
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

            steps_counter += 1
            if (steps_counter > 100):
                break
            if terminated and reward == 0:  # term for failing
                steps_counter = 100

        agent.decay_epsilon()

        eps_rewards_list.append(total_rewards_eps)
        steps_list.append(steps_counter)

    return eps_rewards_list, steps_list


# Function for making the average over a defined amount
def average_over_groups(RewData, group_size=100):
    averages = [] # Initialize list

    for i in range(0, len(RewData), group_size):
        group = RewData[i:(i-1) + group_size]
        averages.append(np.mean(group))

    return averages


# Function for plotting the results
def plot_results(reward_set1, reward_set2, reward_set3, 
                 steps_set1, steps_set2, steps_set3, group_size=100):
    
    fig, ax = plt.subplots(2,3, figsize=(10, 6))
    fig.suptitle('Hyper-Parameters Tuning', fontsize=16)

    plt.subplot(2, 3, 1)
    plt.title('Learning Rate = 0.4,\nEpsilon decay = 0.3')
    plt.ylabel('Rewards (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(reward_set1))
    plt.grid()

    plt.subplot(2, 3, 4)
    #plt.title('Learning Rate = 0.8')
    plt.ylabel('Stpes (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(steps_set1))
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.title('Learning Rate = 0.6,\nEpsilon decay = 0.9')
    plt.ylabel('Rewards (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(reward_set2), color='r')
    plt.grid()

    plt.subplot(2, 3, 5)
    #plt.title('Learning Rate = 0.5')
    plt.ylabel('Stpes (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(steps_set2), color='r')
    plt.grid()
    
    plt.subplot(2, 3, 3)
    plt.title('Learning Rate = 0.8,\nEpsilon decay = 0.5')
    plt.ylabel('Rewards (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(reward_set3), color='g')
    plt.grid()

    plt.subplot(2, 3, 6)
    #plt.title('Learning Rate = 0.2')
    plt.ylabel('Stpes (Avg of 100 episods)')
    plt.xlabel('Episode')
    plt.plot(np.arange(0, n_episodes, group_size), average_over_groups(steps_set3), color='g')
    plt.grid()

    plt.tight_layout()
    plt.show()


# Define number of episodes
n_episodes = 5000
group_size = 100

# Create a new game environment
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")

# Define agents with different hyper-parameters
agent1 = FrozenAgent(
    env=env,
    learning_rate=0.4,
    initial_epsilon = 0.8,
    epsilon_decay = 0.3 / (n_episodes / 2),  # reduce the exploration over time
    final_epsilon=0.1
)
agent2 = FrozenAgent(
    env=env,
    learning_rate=0.6,
    initial_epsilon = 0.8,
    epsilon_decay = 0.9 / (n_episodes / 2),  # reduce the exploration over time
    final_epsilon=0.1
)
agent3 = FrozenAgent(
    env=env,
    learning_rate=0.8,
    initial_epsilon = 0.8,
    epsilon_decay = 0.5 / (n_episodes / 2),  # reduce the exploration over time
    final_epsilon=0.1
)

# Running the agents over the game
agent1_rewards, agent1_steps = run_game(agent1)
agent2_rewards, agent2_steps = run_game(agent2)
agent3_rewards, agent3_steps = run_game(agent3)

plot_results(agent1_rewards, agent2_rewards, agent3_rewards, agent1_steps, agent2_steps, agent3_steps)




