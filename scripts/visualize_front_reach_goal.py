import gymnasium as gym
import numpy as np
from tqdm import tqdm
import env # noqa: F401, registers the envs on import
import matplotlib.pyplot as plt

env_config = {
    "id": "goal-safe-v0",
    "max_episode_steps": 20,
    "grid_size": 5,
    "num_humans": 5,
    "help_reward": 1,
    "step_penalty": 0.1,
    "terminal_reward": 1,
    "proximity_reward": 0.05,
    "obs_as_grid": False
}

env_= gym.make(**env_config)

# create random transitions and visualize the pareto front
# create 1000 random rollouts
num_episodes = 10000
transitions = []


eval_returns = np.zeros((num_episodes, 2), dtype=np.float32)
for episode in tqdm(range(num_episodes)):
    prev_action = None
    ep_returns = []
    obs, _ = env_.reset()
    done = False
    while not done:
        # sample until we get a different action than the previous one, to encourage more diverse transitions
        action = env_.action_space.sample()
        while action == prev_action:
            action = env_.action_space.sample()
        prev_action = action
        next_obs, reward, terminated, truncated, info = env_.step(action)
        done = terminated or truncated
        transitions.append((obs, reward, next_obs, done))
        obs = next_obs
        ep_returns.append(reward)
    eval_returns[episode] = np.mean(ep_returns, axis=0)

def visualize_pareto_front(rewards):

    # Compute the Pareto front
    pareto_mask = np.ones(len(rewards), dtype=bool)
    for i in range(len(rewards)):
        if not pareto_mask[i]:
            continue
        dominated = (
            (rewards[:, 0] <= rewards[i, 0]) &
            (rewards[:, 1] <= rewards[i, 1]) &
            ((rewards[:, 0] < rewards[i, 0]) | (rewards[:, 1] < rewards[i, 1]))
        )
        dominated[i] = False
        pareto_mask &= ~dominated

    pareto_rewards = rewards[pareto_mask]

    # Plot the Pareto front
    plt.figure(figsize=(8, 6))
    plt.scatter(rewards[:, 0], rewards[:, 1], alpha=0.3, label="All Transitions")
    plt.scatter(pareto_rewards[:, 0], pareto_rewards[:, 1], color='red', label="Pareto Front")
    plt.xlabel("Task Return")
    plt.ylabel("Help Return")
    plt.title("Pareto Front of Task vs Help Returns")
    plt.legend()
    plt.grid()
    plt.show()

visualize_pareto_front(eval_returns)