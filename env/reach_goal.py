from enum import IntEnum
from typing import Any
import gymnasium as gym
from numpy import float32
import numpy as np
import pygame


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ReachGoalEnv(gym.Env):
    def __init__(
        self,
        grid_size=10,
        num_humans=20,
        step_penalty=0.4,
        terminal_reward=2,
        obs_as_grid=True,
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.obs_as_grid = obs_as_grid
        self.action_space = gym.spaces.Discrete(len(Action))
        if obs_as_grid:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(4 + self.num_humans * 4,), dtype=float32
            )

        self.reward_space = gym.spaces.Box(
            low=np.array([-step_penalty, 0.0], dtype=float32),
            high=np.array([terminal_reward - step_penalty, 1], dtype=float32),
        )

        self.help_reward = 1
        self.goal_pos = [0, self.grid_size - 1]  # top right
        self.agent_pos = None
        self.human_ages = None
        self.human_positions = None
        self.helped = None
        self.reset()


    def reset(self,seed=None, options=None):
        # top left
        self.agent_pos = [0, 0]
        self.human_ages = np.random.uniform(0.001, 1.0, self.num_humans).round(2)
        # random positions in a normal distribution around the top middle of the grid
        self.human_positions = self.distribute_humans(self.grid_size, self.num_humans)
        self.human_positions = np.array(sorted(self.human_positions, key=lambda pos: (pos[0], pos[1])))  # sort by row, then column for consistency
        self.helped = np.zeros(len(self.human_positions), dtype=bool)
        return self.get_obs(), {}

    def get_obs(self):
        if self.obs_as_grid:
            return self.get_obs_grid()
        else:
            return self.get_obs_vector()

    def get_obs_grid(self):
        """Get the observation as a 3D tensor for CNN input, with separate channels for agent, goal"""
        # Channel 0: agent, Channel 1: goal, Channel 2: human urgency
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=float32)
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        obs[1, self.goal_pos[0], self.goal_pos[1]] = 1.0

        rows = self.human_positions[:, 0]
        cols = self.human_positions[:, 1]
        urgency = np.where(self.helped, 0.0, self.human_ages)
        obs[2, rows, cols] = urgency

        return obs

    def get_obs_vector(self):
        """
        Concatenate the agent position, goal position, and human positions into a single observation vector.
        """
        obs = np.zeros((2 + 2 + self.num_humans * 4), dtype=float32)
        obs[0] = self.agent_pos[0] / self.grid_size
        obs[1] = self.agent_pos[1] / self.grid_size
        obs[2] = self.goal_pos[0] / self.grid_size
        obs[3] = self.goal_pos[1] / self.grid_size
        # human positions, ages, and helped statuses
        obs[4 : 4 + self.num_humans * 2] = (
            self.human_positions.flatten() / self.grid_size
        )
        obs[4 + self.num_humans * 2 : 4 + self.num_humans * 3] = self.human_ages
        obs[4 + self.num_humans * 3 :] = self.helped.astype(float32)
        return obs

    def normal_curve(self, x, grid_size, floor=2.0, sigma=None):
        if sigma is None:
            sigma = max(grid_size / 4, 0.5)
        center = grid_size / 2
        peak = max(floor + 0.5, grid_size - 1)
        return floor + (peak - floor) * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def distribute_humans(self, grid_size, num_humans, floor=2.0):
        """
        Distribute humans in a normal distribution around the top middle of the grid, with more space at the bottom.
        Throws an error if num_humans is greater than the number of valid positions under the curve.
        """

        # All valid x columns and their max y under the curve
        xs = np.arange(grid_size)
        upper = self.normal_curve(xs, grid_size, floor=floor)
        y_min = int(round(floor))
        y_max = np.clip(np.round(upper).astype(int), y_min, grid_size - 1)

        # Count of valid y's per column, then build (y, x) for every point
        counts = np.maximum(y_max - y_min + 1, 0)
        cols = np.repeat(xs, counts)
        # classic "ranges from counts" trick: 0..c0-1, 0..c1-1, ... then shift by y_min
        rows = (
            np.arange(counts.sum())
            - np.repeat(np.cumsum(counts) - counts, counts)
            + y_min
        )
        all_points = np.column_stack([rows, cols])
        if num_humans > len(all_points):
            raise ValueError(
                f"num_humans {num_humans} is greater than the number of valid positions {len(all_points)} under the curve. Reduce num_humans or increase grid_size."
            )

        # Pick num_humans of them without replacement
        n = min(num_humans, len(all_points))
        idx = np.random.choice(len(all_points), size=n, replace=False)
        return all_points[idx]

    def step(self, action):
        x_pos, y_pos = self.agent_pos

        # perform the movement
        if action == Action.UP:
            x_pos -= 1
        elif action == Action.RIGHT:
            y_pos += 1
        elif action == Action.DOWN:
            x_pos += 1
        elif action == Action.LEFT:
            y_pos -= 1
        else:
            raise Exception(f"bad action {action}")

        terminated = False
        reward = np.zeros(2, dtype=np.float32)
        reward[0] -= self.step_penalty

        # out of bounds, cannot move
        if y_pos < 0 or y_pos >= self.grid_size or x_pos < 0 or x_pos >= self.grid_size:
            return (
                self.get_obs(),
                reward,
                terminated,
                False,
                {},
            )

        # into a blocked cell, cannot move
        self.agent_pos = [x_pos, y_pos]

        # into a human cell
        human_matches = np.where((self.human_positions == self.agent_pos).all(axis=1))[
            0
        ]
        if len(human_matches) > 0:
            human_idx = human_matches[0]
            if not self.helped[human_idx]:
                self.helped[human_idx] = True
                reward[1] += self.help_reward

        # into goal cell
        if self.agent_pos == self.goal_pos:
            reward[0] += self.terminal_reward
            terminated = True

        return (
            self.get_obs(),
            reward,
            terminated,
            False,
            {},
        )
