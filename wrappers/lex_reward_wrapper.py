"""Reward wrapper for lexicographic triangle-vs-circle optimization.

The MyFourRoom environment emits a 4-dimensional reward vector:
    [blue_triangle, blue_circle, red_triangle, red_circle]

This wrapper collapses it to a 2-dimensional lexicographic reward:
    obj[0] = triangle score  (blue_triangle + red_triangle)
    obj[1] = circle  score   (blue_circle  + red_circle)

When ``triangle_first=False`` the order is swapped:
    obj[0] = circle  score
    obj[1] = triangle score

The resulting reward is passed to any lexicographic agent as a
(reward_size=2) vector where index 0 always has the highest priority.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box


class LexRewardWrapper(gym.RewardWrapper):
    """Combine the 4-dim FourRoom reward into 2-dim lex objectives.

    Parameters
    ----------
    env : gym.Env
        The wrapped environment (must have a 4-dim reward space).
    triangle_first : bool
        If True  → [triangle, circle]  (triangle is prioritised).
        If False → [circle,  triangle] (circle   is prioritised).
    """

    def __init__(self, env: gym.Env, triangle_first: bool = True):
        super().__init__(env)
        self.triangle_first = triangle_first

        # Override the reward space metadata so that lex agents can read it.
        # Do NOT touch reward_dim: the base env uses it internally to size
        # the zero-reward vectors it emits.
        self.env.unwrapped.reward_space = Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.full(2, 4.0, dtype=np.float32),
            dtype=np.float32,
        )

    def reward(self, reward: np.ndarray) -> np.ndarray:
        """Map 4D → 2D reward.

        Parameters
        ----------
        reward : np.ndarray, shape (4,)
            [blue_triangle, blue_circle, red_triangle, red_circle]

        Returns
        -------
        np.ndarray, shape (2,)
            [obj0, obj1] where obj0 is the high-priority objective
            (triangle when ``triangle_first=True``, circle otherwise)
            and obj1 is the lower-priority objective.
        """
        triangle = float(reward[0]) + float(reward[2])  # blue_tri + red_tri
        circle   = float(reward[1]) + float(reward[3])  # blue_circ + red_circ
        if self.triangle_first:
            return np.array([triangle, circle], dtype=np.float32)
        else:
            return np.array([circle, triangle], dtype=np.float32)
