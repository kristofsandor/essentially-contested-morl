from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Convert2DStateWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that converts a 4D vector observation
    [row, col, has_gold, has_diamond] into a single Discrete index
    over a 2D meta-grid suitable for PQL.

    Meta-grid layout:
        - Base grid: rows x cols (agent position).
        - Meta-rows: 2 * rows  (top: no gold, bottom: has gold).
        - Meta-cols: 2 * cols  (left: no diamond, right: has diamond).

    Mapping:
        meta_row = row + has_gold * rows
        meta_col = col + has_diamond * cols
        index    = meta_row * (2 * cols) + meta_col
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        rows, cols = self._infer_grid_shape(env)
        self._rows = int(rows)
        self._cols = int(cols)

        # 2d location on the grid
        self.observation_space = spaces.MultiDiscrete([2 * self._rows, 2 * self._cols])

    @staticmethod
    def _infer_grid_shape(env: gym.Env) -> Tuple[int, int]:
        """
        Infer the underlying grid shape from the wrapped environment.

        Preference:
            1. Use env.map.shape if available.
            2. Fall back to env.size assuming a square grid.
        """
        if hasattr(env.unwrapped, "map") and isinstance(getattr(env.unwrapped, "map"), np.ndarray):
            shape = env.unwrapped.map.shape
            if len(shape) != 2:
                raise ValueError(f"Expected 2D map, got shape {shape}")
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
        Convert [row, col, has_gold, has_diamond] into a single index.
        """
        row = int(observation[0])
        col = int(observation[1])
        has_gold = int(observation[2])
        has_diamond = int(observation[3])

        meta_row = row + has_gold * self._rows
        meta_col = col + has_diamond * self._cols

        return np.array([meta_row, meta_col])
