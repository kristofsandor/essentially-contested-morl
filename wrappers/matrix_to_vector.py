import gymnasium as gym
import numpy as np


class DeontologicalWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(low=rs.low[0], high=rs.high[0])

    def reward(self, r):
        return r[0].astype(np.float32)


class UtilitarianWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=np.array([rs.low[0, 0], rs.low[1, 1]], dtype=np.float32),
            high=np.array([rs.high[0, 0], rs.high[1, 1]], dtype=np.float32),
        )

    def reward(self, r):
        return np.array([r[0, 0], r[1, 1]], dtype=np.float32)
