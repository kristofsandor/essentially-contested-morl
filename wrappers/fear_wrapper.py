import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FearWrapper(gym.RewardWrapper):

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.reward_space = Box(
            low=np.array([-1.0, -0.5, 0.0, 0.0]),
            high=np.array([0.0, 0.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32,
        )

    def reward(self, reward):
        current_pos = self.env.unwrapped.current_pos
        cell = self.env.unwrapped.get_map_value(current_pos)

        if cell == "E1" or cell == "E2" and reward[0] != -1:
            fear_reward = -0.5
        else:
            fear_reward = 0
        reward = np.insert(reward, 1, fear_reward)

        return reward