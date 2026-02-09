from enum import Enum
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class MultiObjectiveGridWorldEnv(gym.Env):
    """
    Multi-objective grid world environment with multiple targets/regions.
    The agent needs to serve multiple regions, and we track:
    - Task reward: reaching targets
    - Fairness metrics: equity vs utility
    - Efficiency metrics: time/distance to serve regions
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, num_regions=3, max_steps=100):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.num_regions = num_regions
        self.max_steps = max_steps
        self.step_count = 0

        # Observations include agent location and all region locations
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "regions": spaces.Box(0, size - 1, shape=(num_regions, 2), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # Track objective metrics
        self.region_visits = np.zeros(
            num_regions, dtype=int
        )  # How many times each region was visited
        self.region_last_visit_step = np.full(
            num_regions, -1, dtype=int
        )  # Last step each region was visited
        self.region_distances = np.zeros(num_regions)  # Distance to each region
        self.episode_rewards_per_region = np.zeros(
            num_regions
        )  # Rewards accumulated per region

    def _get_obs(self):
        return {"agent": self._agent_location, "regions": self._region_locations}

    def _get_info(self):
        """Return comprehensive info about objectives"""
        # Calculate distances to all regions
        distances = [
            np.linalg.norm(self._agent_location - region, ord=1)
            for region in self._region_locations
        ]

        # Calculate fairness metrics
        visit_counts = self.region_visits.copy()
        visit_proportions = visit_counts / (visit_counts.sum() + 1e-8)

        # Equity: minimize variance in visit counts (more equal = better)
        equity = -np.var(visit_proportions) if visit_counts.sum() > 0 else 0

        # Utility: maximize total visits (more visits = better)
        utility = visit_counts.sum()

        # Efficiency: minimize average distance to regions
        efficiency = -np.mean(distances) if len(distances) > 0 else 0

        return {
            "distances": distances,
            "region_visits": visit_counts.copy(),
            "region_proportions": visit_proportions.copy(),
            "equity": equity,
            "utility": utility,
            "efficiency": efficiency,
            "task_reward": self._task_reward,
            "step": self.step_count,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place regions randomly, ensuring they don't coincide with agent
        self._region_locations = []
        for _ in range(self.num_regions):
            region_loc = self._agent_location
            while np.array_equal(region_loc, self._agent_location) or any(
                np.array_equal(region_loc, r) for r in self._region_locations
            ):
                region_loc = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._region_locations.append(region_loc)
        self._region_locations = np.array(self._region_locations)

        # Reset objective tracking
        self.region_visits = np.zeros(self.num_regions, dtype=int)
        self.region_last_visit_step = np.full(self.num_regions, -1, dtype=int)
        self.region_distances = np.zeros(self.num_regions)
        self.episode_rewards_per_region = np.zeros(self.num_regions)
        self._task_reward = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self.step_count += 1

        # Map the action to direction
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached any region
        self._task_reward = 0
        reached_regions = []

        for i, region_loc in enumerate(self._region_locations):
            if np.array_equal(self._agent_location, region_loc):
                self.region_visits[i] += 1
                self.region_last_visit_step[i] = self.step_count
                self._task_reward += 1  # Task reward for reaching a region
                reached_regions.append(i)
                self.episode_rewards_per_region[i] += 1

        # Episode terminates if all regions visited or max steps reached
        terminated = (self.region_visits.min() > 0) or (
            self.step_count >= self.max_steps
        )
        truncated = self.step_count >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        info["reached_regions"] = reached_regions

        if self.render_mode == "human":
            self._render_frame()

        return observation, self._task_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw regions with different colors based on visit count
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]
        for i, region_loc in enumerate(self._region_locations):
            color = colors[i % len(colors)]
            # Darker color if visited more
            visit_factor = min(self.region_visits[i] / 5.0, 1.0)
            color = tuple(int(c * (0.5 + 0.5 * visit_factor)) for c in color)

            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    pix_square_size * region_loc,
                    (pix_square_size, pix_square_size),
                ),
            )

            # Draw visit count
            if self.region_visits[i] > 0:
                font = pygame.font.Font(None, 24)
                text = font.render(str(self.region_visits[i]), True, (255, 255, 255))
                canvas.blit(text, pix_square_size * region_loc + (5, 5))

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
