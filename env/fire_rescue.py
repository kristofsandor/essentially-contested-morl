"""
Fire Rescue Gridworld Environment for Multi-Objective RL that considers different interpretations of values.

This environment implements a two-level structure:
- Level 1: Values (Safety, Fairness) + Task Reward
- Level 2: Different ethical theories interpreting each value:
  * Safety: Sentient Utilitarianism, Classical Utilitarianism, Hedonistic Utilitarianism
  * Fairness: Equal help, Proportional to need, Minimum threshold

The environment features:
- Fire spreading probabilistically in a gridworld
- Humans and dogs of different ages with varying vulnerability levels
- Collectible diamonds scattered throughout the grid
- Agent must collect diamonds (task reward) and rescue entities before they die from fire damage
- Different ethical interpretations of what counts as "safe" and "fair"
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec


class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    RESCUE = 4
    COLLECT = 5


class EntityType(Enum):
    HUMAN = 0
    DOG = 1
    NONE = 2


class FireRescueEnv(gym.Env):
    """
    Fire Rescue Gridworld Environment with Multi-Objective Rewards.

    The agent navigates a gridworld to:
    1. Collect diamonds for task reward
    2. Rescue entities from spreading fire
    3. Maximize safety according to different utilitarian theories
    4. Ensure fairness in help distribution

    Safety rewards (three utilitarian theories):
    - Sentient Utilitarianism: maximize all lives (humans + dogs)
    - Classical Utilitarianism: maximize human lives only
    - Hedonistic Utilitarianism: maximize quality-adjusted life years

    Fairness rewards (three interpretations):
    - Equal help: no one got more help than others
    - Proportional to need: help proportionate to vulnerability
    - Minimum threshold: everyone got some help

    Reward vector structure:
    [task_reward,
     safety_sentient, safety_classical, safety_hedonistic,
     fairness_equal, fairness_proportional, fairness_minimum]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size=10,
        num_humans=5,
        num_dogs=3,
        num_diamonds=8,
        max_steps=100,
        fire_spread_prob=0.3,
        initial_fire_cells=3,
        diamond_reward=0.1,
    ):
        """
        Args:
            size: Grid size (size x size)
            num_humans: Number of humans to rescue
            num_dogs: Number of dogs to rescue
            num_diamonds: Number of diamonds to collect
            max_steps: Maximum steps per episode
            fire_spread_prob: Probability of fire spreading to adjacent cell each step
            initial_fire_cells: Number of initial fire cells
            diamond_reward: Reward per collected diamond
        """
        self.size = size
        self.window_size = 600
        self.num_humans = num_humans
        self.num_dogs = num_dogs
        self.num_entities = num_humans + num_dogs
        self.num_diamonds = num_diamonds
        self.diamond_reward = diamond_reward
        self.max_steps = max_steps
        self.fire_spread_prob = fire_spread_prob
        self.initial_fire_cells = initial_fire_cells
        self.step_count = 0

        # Observation space: agent position + fire map + entity information + diamonds
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "fire": spaces.MultiBinary(size * size),  # Full fire map (flattened)
                "entities": spaces.Dict(
                    {
                        "positions": spaces.Box(
                            0, size - 1, shape=(self.num_entities, 2), dtype=int
                        ),
                        "types": spaces.MultiBinary(
                            self.num_entities
                        ),  # 0=human, 1=dog
                        "ages": spaces.Box(
                            0, 100, shape=(self.num_entities,), dtype=int
                        ),
                        "vulnerability": spaces.Box(
                            0.0, 1.0, shape=(self.num_entities,), dtype=float
                        ),
                        "rescued": spaces.MultiBinary(self.num_entities),
                        "damage": spaces.Box(
                            0, 5, shape=(self.num_entities,), dtype=int
                        ),
                    }
                ),
                "diamonds": spaces.Dict(
                    {
                        "positions": spaces.Box(
                            0, size - 1, shape=(self.num_diamonds, 2), dtype=int
                        ),
                        "collected": spaces.MultiBinary(self.num_diamonds),
                    }
                ),
            }
        )

        # Action space: 4 movement + 1 rescue + 1 collect
        self.action_space = spaces.Discrete(6)

        # Reward space: 7-dimensional vector
        # [task_reward, safety_sentient, safety_classical, safety_hedonistic,
        #  fairness_equal, fairness_proportional, fairness_minimum]
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )
        self.reward_dim = 7

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
            Actions.RESCUE.value: np.array([0, 0]),
            Actions.COLLECT.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Set spec if not already set (e.g., when instantiated directly vs via gymnasium.make)
        if self.spec is None:
            try:
                # Try to get spec from registry
                from gymnasium.envs.registration import registry
                if "FireRescue-v0" in registry:
                    self.spec = registry["FireRescue-v0"]
                else:
                    # Create a minimal spec if not in registry
                    self.spec = EnvSpec(
                        id="FireRescue-v0",
                        entry_point="env.fire_rescue_env:FireRescueEnv",
                        max_episode_steps=max_steps,
                    )
            except Exception:
                # Fallback: create a simple spec-like object
                class SimpleSpec:
                    def __init__(self, env_id):
                        self.id = env_id
                self.spec = SimpleSpec("FireRescue-v0")

        self.window = None
        self.clock = None

        # Entity tracking - all numpy arrays for vectorization
        self.entity_types = np.zeros(self.num_entities, dtype=int)  # 0=human, 1=dog
        self.entity_ages = np.zeros(
            self.num_entities, dtype=int
        )  # Ages: 0-100 for humans, 0-15 for dogs
        self.entity_vulnerability = np.zeros(
            self.num_entities, dtype=float
        )  # Vulnerability levels 0-1
        self.entity_rescued = np.zeros(self.num_entities, dtype=bool)
        self.entity_fire_damage = np.zeros(self.num_entities, dtype=int)
        self.entity_help_received = np.zeros(self.num_entities, dtype=int)
        self.entity_positions = np.full(
            (self.num_entities, 2), 0, dtype=int
        ) 
        self.entity_alive = np.ones(self.num_entities, dtype=bool)

        # Fire tracking - boolean grid for O(1) lookup and vectorization
        self.fire_positions = np.zeros(
            (self.size, self.size), dtype=bool
        )  # Boolean grid (row, col)

        # Diamond tracking
        self.diamond_positions = np.zeros((self.num_diamonds, 2), dtype=int)
        self.diamond_collected = np.zeros(self.num_diamonds, dtype=bool)

        # Agent position
        self._agent_location = np.array([0, 0])

    def _get_obs(self):
        """Get current observation."""
        # Create fire map (flattened grid) - vectorized from boolean grid
        fire_map = self.fire_positions.flatten().astype(int)

        return {
            "agent": self._agent_location.copy(),
            "fire": fire_map,
            "entities": {
                "positions": self.entity_positions.copy(),
                "types": self.entity_types.copy(),
                "ages": self.entity_ages.copy(),
                "vulnerability": self.entity_vulnerability.copy(),
                "rescued": self.entity_rescued.copy(),
                "damage": self.entity_fire_damage.copy(),
            },
            "diamonds": {
                "positions": self.diamond_positions.copy(),
                "collected": self.diamond_collected.copy(),
            },
        }

    def _spread_fire(self):
        """Spread fire probabilistically to adjacent cells."""

        # Check 4 directions for fire spread
        neighbors = np.zeros_like(self.fire_positions)
        neighbors[1:, :] |= self.fire_positions[:-1, :]
        neighbors[:-1, :] |= self.fire_positions[1:, :]
        neighbors[:, 1:] |= self.fire_positions[:, :-1]
        neighbors[:, :-1] |= self.fire_positions[:, 1:]

        # Find new fire positions (neighbors that aren't already on fire)
        prob_mask = (
            self.np_random.random((self.size, self.size)) < self.fire_spread_prob
        )
        self.fire_positions |= neighbors & prob_mask

    def _apply_fire_damage(self):
        """Apply fire damage to entities in fire cells."""
        # Get valid entities (not rescued and alive)
        valid_mask = ~self.entity_rescued & self.entity_alive
        entities_in_fire = self.fire_positions[
            self.entity_positions[:, 0], self.entity_positions[:, 1]
        ]
        self.entity_fire_damage[valid_mask & entities_in_fire] += 1
        self.entity_alive[self.entity_fire_damage >= 5] = False

    def _calculate_safety_rewards(self) -> Tuple[float, float, float]:
        """
        Calculate safety rewards using three utilitarian theories.

        Returns:
            (sentient, classical, hedonistic) safety rewards
        """
        # Sentient Utilitarianism: maximize all lives (humans + dogs)
        rescued_mask = self.entity_rescued
        injured_mask = self.entity_alive & (self.entity_fire_damage > 0) & ~rescued_mask
        sentient_reward = float(
            rescued_mask.sum()
            + (1.0 - self.entity_fire_damage[injured_mask] / 5.0).sum()
        )

        # Classical Utilitarianism: maximize human lives only
        human_mask = self.entity_types == EntityType.HUMAN.value
        rescued_humans = rescued_mask & human_mask
        injured_humans = injured_mask & human_mask
        classical_reward = float(
            rescued_humans.sum()
            + (1.0 - self.entity_fire_damage[injured_humans] / 5.0).sum()
        )

        # Hedonistic Utilitarianism: maximize quality-adjusted life years
        # Calculate remaining life expectancy vectorized
        remaining_years = np.where(
            self.entity_types == EntityType.HUMAN.value,
            np.maximum(0, 80 - self.entity_ages),
            np.maximum(0, 15 - self.entity_ages),
        )
        hedonistic_reward = float(
            remaining_years[rescued_mask].sum()
            + (
                remaining_years[injured_mask]
                * (1.0 - self.entity_fire_damage[injured_mask] / 5.0)
            ).sum()
        )

        return sentient_reward, classical_reward, hedonistic_reward

    def _calculate_fairness_rewards(self) -> Tuple[float, float, float]:
        """
        Calculate fairness rewards using three interpretations.

        Returns:
            (equal, proportional, minimum) fairness rewards
        """
        total_help = self.entity_help_received.sum()
        vulnerability_sum = self.entity_vulnerability.sum()

        # Fairness a) Equal help: no one got more help than others
        help_variance = np.var(self.entity_help_received)
        fairness_equal = -help_variance  # Lower variance = more equal = better

        # Fairness b) Proportional to need: help proportionate to vulnerability
        # Compare distributions of expected vs received help
        if total_help > 0 and vulnerability_sum > 0:
            # Calculate expected help distribution based on vulnerability
            expected_help_dist = (
                self.entity_vulnerability * total_help / vulnerability_sum
            )

            # Normalize to probability distributions for comparison
            received_dist = self.entity_help_received / total_help
            expected_dist = expected_help_dist / total_help

            # Compare distributions using negative KL divergence (higher = more similar)
            epsilon = 1e-10
            received_dist_safe = received_dist + epsilon
            expected_dist_safe = expected_dist + epsilon

            # KL divergence: sum(p * log(p/q))
            fairness_proportional = -np.sum(
                received_dist_safe * np.log(received_dist_safe / expected_dist_safe)
            )
        else:
            fairness_proportional = 0.0

        # Fairness c) Minimum threshold: everyone got some help
        # Reward is the lowest level of help that everyone received
        fairness_minimum = float(self.entity_help_received.min())

        return fairness_equal, fairness_proportional, fairness_minimum

    def _get_info(self) -> Dict:
        """Return comprehensive info about objectives."""
        safety_sent, safety_class, safety_hed = self._calculate_safety_rewards()
        fairness_eq, fairness_prop, fairness_min = self._calculate_fairness_rewards()

        rescued_count = self.entity_rescued.sum()
        injured_count = np.sum(
            (self.entity_alive) & (self.entity_fire_damage > 0) & (~self.entity_rescued)
        )
        dead_count = np.sum(~self.entity_alive)

        entity_types_array = np.array(self.entity_types)
        injured_humans = np.sum(
            (entity_types_array == EntityType.HUMAN.value)
            & (self.entity_alive)
            & (self.entity_fire_damage > 0)
            & (~self.entity_rescued)
        )
        injured_dogs = np.sum(
            (entity_types_array == EntityType.DOG.value)
            & (self.entity_alive)
            & (self.entity_fire_damage > 0)
            & (~self.entity_rescued)
        )

        # Calculate diamond collection statistics
        diamonds_collected = int(self.diamond_collected.sum())
        task_reward = diamonds_collected * self.diamond_reward

        return {
            "task_reward": task_reward,
            "safety_sentient": safety_sent,
            "safety_classical": safety_class,
            "safety_hedonistic": safety_hed,
            "fairness_equal": fairness_eq,
            "fairness_proportional": fairness_prop,
            "fairness_minimum": fairness_min,
            "rescued_count": rescued_count,
            "injured_count": injured_count,
            "dead_count": dead_count,
            "injured_humans": injured_humans,
            "injured_dogs": injured_dogs,
            "diamonds_collected": diamonds_collected,
            "diamonds_total": self.num_diamonds,
            "fire_cells": int(self.fire_positions.sum()),
            "step": self.step_count,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset entity tracking - initialize as numpy arrays
        self.entity_types = np.zeros(self.num_entities, dtype=int)
        self.entity_ages = np.zeros(self.num_entities, dtype=int)
        self.entity_vulnerability = np.zeros(self.num_entities, dtype=float)
        self.entity_rescued = np.zeros(self.num_entities, dtype=bool)
        self.entity_fire_damage = np.zeros(self.num_entities, dtype=int)
        self.entity_help_received = np.zeros(self.num_entities, dtype=int)
        self.entity_positions = np.full((self.num_entities, 2), -1, dtype=int)
        self.entity_alive = np.ones(self.num_entities, dtype=bool)

        # Reset fire tracking
        self.fire_positions = np.zeros((self.size, self.size), dtype=bool)

        # Randomly place agent
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Fill entity arrays vectorized
        self.entity_types[: self.num_humans] = EntityType.HUMAN.value  # Humans
        self.entity_types[self.num_humans :] = EntityType.DOG.value  # Dogs
        self.entity_ages[: self.num_humans] = self.np_random.integers(
            0, 101, size=self.num_humans
        )
        self.entity_ages[self.num_humans :] = self.np_random.integers(
            0, 16, size=self.num_dogs
        )
        self.entity_vulnerability[:] = self.np_random.random(self.num_entities)

        # Reset diamond tracking
        self.diamond_collected = np.zeros(self.num_diamonds, dtype=bool)

        # Single random draw over 2D grid: assign positions to entities, diamonds, and fire
        total_positions_needed = (
            self.num_entities + self.num_diamonds + self.initial_fire_cells
        )
        total_grid_cells = self.size * self.size

        if total_positions_needed > total_grid_cells:
            raise ValueError(
                f"Not enough grid cells for all objects. "
                f"Need {total_positions_needed} positions but grid only has {total_grid_cells} cells."
            )

        # Randomly select positions from the entire grid without replacement
        selected_indices = self.np_random.choice(
            total_grid_cells, size=total_positions_needed, replace=False
        )

        # Convert flat indices to (row, col) coordinates
        selected_positions = np.array(
            [(idx // self.size, idx % self.size) for idx in selected_indices]
        )

        # Assign positions: first num_entities to entities, next num_diamonds to diamonds, rest to fire
        self.entity_positions = selected_positions[: self.num_entities]
        self.diamond_positions = selected_positions[
            self.num_entities : self.num_entities + self.num_diamonds
        ]
        fire_positions_flat = selected_indices[self.num_entities + self.num_diamonds :]

        # Initialize fire positions - set in 2d boolean grid
        self.fire_positions.flat[fire_positions_flat] = True

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _self_rescue(self):
        """Self-rescue all unrescued entities based on vulnerability."""
        self_rescue_probs = (
            1.0 - self.entity_vulnerability
        ) * 0.05  # 0-5% chance based on vulnerability
        self_rescued = self.np_random.random(self.num_entities) < self_rescue_probs
        self.entity_rescued[self_rescued] = True

    def _check_termination(self):
        """Check termination and truncation."""
        terminated = (self.entity_rescued | ~self.entity_alive).all()
        truncated = self.step_count >= self.max_steps  # Time limit reached
        return terminated, truncated

    def _handle_movement(self, action):
        """Handle movement."""
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

    def _handle_rescue(self):
        """Handle rescue."""
        # Find entity at current position - vectorized
        position_match = (self.entity_positions == self._agent_location).all(axis=1)
        valid_entities = position_match & ~self.entity_rescued & self.entity_alive
        # Get first matching entity index, or None if no match
        rescued_entity_idx = (
            np.flatnonzero(valid_entities)[0] if valid_entities.any() else None
        )
        if rescued_entity_idx is not None:
            # Track rescue attempt
            self.entity_help_received[rescued_entity_idx] += 1

            # Rescue success depends on vulnerability
            vulnerability = self.entity_vulnerability[rescued_entity_idx]
            success_prob = 1.0 - vulnerability

            if self.np_random.random() < success_prob:
                # Successful rescue
                self.entity_rescued[rescued_entity_idx] = True
                return True

        return False

    def _handle_collect(self):
        """Handle diamond collection."""
        # Find diamond at current position - vectorized
        position_match = (self.diamond_positions == self._agent_location).all(axis=1)
        uncollected_mask = ~self.diamond_collected
        collectable = position_match & uncollected_mask

        if collectable.any():
            idx = np.flatnonzero(collectable)[0]
            self.diamond_collected[idx] = True
            return True
        return False

    def step(self, action):
        """Execute one step in the environment."""

        # Initialize reward vector: [task, safety_sent, safety_class, safety_hed, fairness_eq, fairness_prop, fairness_min]
        reward_vector = np.zeros(7)

        self.step_count += 1

        # Track diamonds collected before action
        diamonds_collected_before = self.diamond_collected.sum()

        # Handle movement, rescue, or collect action
        if action == Actions.RESCUE.value:
            self._handle_rescue()
        elif action == Actions.COLLECT.value:
            self._handle_collect()
        else:  # Movement
            self._handle_movement(action)

        # Track diamonds collected after action
        diamonds_collected_after = self.diamond_collected.sum()
        diamonds_collected_this_step = (
            diamonds_collected_after - diamonds_collected_before
        )

        # Calculate task reward based on diamonds collected
        reward_vector[0] = diamonds_collected_this_step * self.diamond_reward

        # Spread fire
        self._spread_fire()

        # Self-rescue: all unrescued entities have a chance to self-rescue based on vulnerability
        self._self_rescue()

        # Apply fire damage to entities in fire
        self._apply_fire_damage()

        # Calculate safety and fairness rewards
        safety_sent, safety_class, safety_hed = self._calculate_safety_rewards()
        fairness_eq, fairness_prop, fairness_min = self._calculate_fairness_rewards()

        reward_vector[1] = safety_sent
        reward_vector[2] = safety_class
        reward_vector[3] = safety_hed
        reward_vector[4] = fairness_eq
        reward_vector[5] = fairness_prop
        reward_vector[6] = fairness_min

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        terminated, truncated = self._check_termination()

        return observation, reward_vector, terminated, truncated, info

    def _render_frame(self):
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((240, 240, 240))  # Light gray background
        pix_square_size = self.window_size / self.size

        # Draw fire cells (red/orange) - iterate only over True positions
        fire_coords = np.argwhere(self.fire_positions)
        fire_color = (255, 100, 0) if (self.step_count % 2 == 0) else (255, 150, 50)
        for row, col in fire_coords:
            pygame.draw.rect(
                canvas,
                fire_color,
                pygame.Rect(
                    pix_square_size * np.array([row, col]),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw diamonds (before entities so they appear below)
        diamond_color = (64, 224, 208)  # Turquoise/cyan color
        for i in range(self.num_diamonds):
            if not self.diamond_collected[i]:  # Only draw uncollected diamonds
                pos = self.diamond_positions[i]
                center = (np.array(pos) + 0.5) * pix_square_size
                # Draw diamond shape (rotated square)
                size = pix_square_size * 0.3
                diamond_points = [
                    (center[0], center[1] - size),  # Top
                    (center[0] + size, center[1]),  # Right
                    (center[0], center[1] + size),  # Bottom
                    (center[0] - size, center[1]),  # Left
                ]
                pygame.draw.polygon(canvas, diamond_color, diamond_points)
                # Add a highlight
                highlight_points = [
                    (center[0], center[1] - size * 0.7),
                    (center[0] + size * 0.5, center[1]),
                    (center[0], center[1] + size * 0.3),
                    (center[0] - size * 0.5, center[1]),
                ]
                pygame.draw.polygon(canvas, (255, 255, 255), highlight_points)

        # Draw entities 
        for i in range(self.num_entities):
            center = (np.array(self.entity_positions[i]) + 0.5) * pix_square_size

            # Determine entity color based on type and state
            if self.entity_rescued[i]:
                # Rescued: green border
                base_color = (100, 200, 100)
            elif not self.entity_alive[i]:
                # Dead: gray (don't apply damage darkening)
                base_color = (100, 100, 100)
                color = base_color
            elif self.entity_types[i] == EntityType.HUMAN.value:  # Human
                # Human: color by age (younger = brighter)
                age_factor = 1.0 - (self.entity_ages[i] / 100.0)
                base_color = (
                    int(100 + 155 * age_factor),
                    int(100 + 100 * age_factor),
                    int(200 + 55 * age_factor),
                )
                # Darken based on damage (only for alive entities)
                damage_factor = 1.0 - (self.entity_fire_damage[i] / 5.0) * 0.5
                color = tuple(int(c * damage_factor) for c in base_color)
            else:  # Dog
                # Dog: brown
                base_color = (139, 69, 19)
                # Darken based on damage (only for alive entities)
                damage_factor = 1.0 - (self.entity_fire_damage[i] / 5.0) * 0.5
                color = tuple(int(c * damage_factor) for c in base_color)

            # Draw entity shape
            if self.entity_types[i] == EntityType.HUMAN.value:  # Human: circle
                pygame.draw.circle(
                    canvas, color, center.astype(int), pix_square_size * 0.3
                )
            else:  # Dog: square
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        center - pix_square_size * 0.25,
                        (pix_square_size * 0.5, pix_square_size * 0.5),
                    ),
                )

            # Draw rescue border if rescued
            if self.entity_rescued[i]:
                pygame.draw.circle(
                    canvas,
                    (0, 255, 0),
                    center.astype(int),
                    pix_square_size * 0.35,
                    width=3,
                )

            # Draw red circle around dead entities
            if not self.entity_alive[i]:
                pygame.draw.circle(
                    canvas,
                    (255, 0, 0),
                    center.astype(int),
                    pix_square_size * 0.35,
                    width=3,
                )

            # Draw damage indicator (only for alive entities)
            if self.entity_fire_damage[i] > 0 and self.entity_alive[i] and not self.entity_rescued[i]:
                damage_radius = (
                    pix_square_size * 0.15 * (self.entity_fire_damage[i] / 5.0)
                )
                pygame.draw.circle(
                    canvas, (255, 0, 0), center.astype(int), int(damage_radius)
                )

        # Draw agent (blue circle)
        agent_center = (self._agent_location + 0.5) * pix_square_size
        pygame.draw.circle(
            canvas, (0, 0, 255), agent_center.astype(int), pix_square_size * 0.25
        )

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )

        # Draw stats text
        if self.render_mode == "human":
            font = pygame.font.Font(None, 24)
            rescued_count = self.entity_rescued.sum()
            dead_count = (~self.entity_alive).sum()
            injured_count = np.sum(
                (self.entity_alive)
                & (self.entity_fire_damage > 0)
                & (~self.entity_rescued)
            )
            diamonds_collected = int(self.diamond_collected.sum())
            stats = [
                f"Step: {self.step_count}/{self.max_steps}",
                f"Rescued: {rescued_count}/{self.num_entities}",
                f"Injured: {injured_count} | Dead: {dead_count}",
                f"Diamonds: {diamonds_collected}/{self.num_diamonds}",
                f"Fire cells: {self.fire_positions.sum()}",
            ]
            y_offset = 10
            for stat in stats:
                text = font.render(stat, True, (0, 0, 0))
                canvas.blit(text, (10, y_offset))
                y_offset += 25

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
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
