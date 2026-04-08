import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class CombineWrapper(gym.RewardWrapper):

    def __init__(
        self,
        env: gym.Env,
        weight,
    ):
        super().__init__(env)
        self.env = env
        self.env.unwrapped.reward_space = Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.weight = weight

    def reward(self, reward):
        # new_reward = np.array([0, 0])
        # harm
        new_reward = np.dot(self.weight, reward.reshape(2,2).T)
        # new_reward[0] = 0.1 * reward[0] + 0.9 * reward[1]
        # # rich
        # new_reward[1] = 0.1 * reward[2] + 0.9 * reward[3]

        return new_reward