from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Convert2DStateWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that converts vector observations into
    a compact 2D state [row, col] suitable for tabular algorithms
    such as PQL.

    This intentionally keeps only the first two coordinates and
    discards additional inventory/features dimensions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        rows, cols = self._infer_grid_shape(env)
        self._rows = int(rows)
        self._cols = int(cols)

        # 2D location on the grid
        self.observation_space = spaces.MultiDiscrete([self._rows, self._cols])

    @staticmethod
    def _infer_grid_shape(env: gym.Env) -> Tuple[int, int]:
        """
        Infer the underlying grid shape from the wrapped environment.

        Preference:
            1. Use env.map.shape if available.
            2. Use env.maze.shape if available.
            3. Fall back to env.size assuming a square grid.
        """
        if hasattr(env.unwrapped, "map") and isinstance(getattr(env.unwrapped, "map"), np.ndarray):
            shape = env.unwrapped.map.shape
            if len(shape) != 2:
                raise ValueError(f"Expected 2D map, got shape {shape}")
            return int(shape[0]), int(shape[1])

        if hasattr(env.unwrapped, "maze") and isinstance(getattr(env.unwrapped, "maze"), np.ndarray):
            shape = env.unwrapped.maze.shape
            if len(shape) != 2:
                raise ValueError(f"Expected 2D maze, got shape {shape}")
            return int(shape[0]), int(shape[1])

        if hasattr(env.unwrapped, "size"):
            size = int(getattr(env.unwrapped, "size"))
            return size, size

        raise ValueError(
            "Could not infer grid shape for Convert2DStateWrapper. "
            "Expected the environment to have either a 2D `map` attribute or a scalar `size`."
        )

    def observation(self, observation: Any):
        """
        Convert any vector-like observation into [row, col].
        """
        row = int(observation[0])
        col = int(observation[1])
        return np.array([row, col], dtype=np.int32)
