from stable_baselines3 import DQN
import gymnasium as gym
import env # noqa: F401, registers the envs on import

from scripts.utils import make_env


class ScalarizeWrapper(gym.RewardWrapper):
    def reward(self, r):
        return float(0.5 * r[0] + 0.5 * r[1])  # task only


env_config = {
    "id": "goal-safe-v0",
    "max_episode_steps": 50,
    "grid_size": 5,
    "num_humans": 5,
    "step_penalty": 0.1,
    "terminal_reward": 1,
    "obs_as_grid": False,
}
env = ScalarizeWrapper(make_env(env_config))
DQN("MlpPolicy", env, verbose=1).learn(50_000)
