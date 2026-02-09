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
- Agent must rescue entities before they die from fire damage
- Different ethical interpretations of what counts as "safe" and "fair"
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3
    rescue = 4


class FireRescueEnv(gym.Env):
    """
    Fire Rescue Gridworld Environment with Multi-Objective Rewards.
    
    The agent navigates a gridworld to:
    1. Rescue entities from spreading fire (task reward)
    2. Maximize safety according to different utilitarian theories
    3. Ensure fairness in help distribution
    
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
        max_steps=100,
        fire_spread_prob=0.3,
        initial_fire_cells=3,
    ):
        """
        Args:
            size: Grid size (size x size)
            num_humans: Number of humans to rescue
            num_dogs: Number of dogs to rescue
            max_steps: Maximum steps per episode
            fire_spread_prob: Probability of fire spreading to adjacent cell each step
            initial_fire_cells: Number of initial fire cells
        """
        self.size = size
        self.window_size = 600
        self.num_humans = num_humans
        self.num_dogs = num_dogs
        self.num_entities = num_humans + num_dogs
        self.max_steps = max_steps
        self.fire_spread_prob = fire_spread_prob
        self.initial_fire_cells = initial_fire_cells
        self.step_count = 0

        # Observation space: agent position + fire map + entity information
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "fire": spaces.MultiBinary(size * size),  # Full fire map (flattened)
                "entities": spaces.Dict({
                    "positions": spaces.Box(0, size - 1, shape=(self.num_entities, 2), dtype=int),
                    "types": spaces.MultiBinary(self.num_entities),  # 0=human, 1=dog
                    "ages": spaces.Box(0, 100, shape=(self.num_entities,), dtype=int),
                    "vulnerability": spaces.Box(0.0, 1.0, shape=(self.num_entities,), dtype=float),
                    "rescued": spaces.MultiBinary(self.num_entities),
                    "damage": spaces.Box(0, 5, shape=(self.num_entities,), dtype=int),
                }),
            }
        )

        # Action space: 4 movement + 1 rescue
        self.action_space = spaces.Discrete(5)

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

        # Entity tracking
        self.entity_types = []  # 0=human, 1=dog
        self.entity_ages = []  # Ages: 0-100 for humans, 0-15 for dogs
        self.entity_vulnerability = []  # Vulnerability levels 0-1
        self.entity_rescued = np.zeros(self.num_entities, dtype=bool)
        self.entity_fire_damage = np.zeros(self.num_entities, dtype=int)
        self.entity_help_received = np.zeros(self.num_entities, dtype=int)
        self.entity_positions = []  # List of (row, col) tuples or None if dead/rescued
        self.entity_alive = np.ones(self.num_entities, dtype=bool)

        # Fire tracking
        self.fire_positions = set()  # Set of (row, col) tuples
        self.steps_since_all_burned = None  # Track steps since everything burned down

        # Agent position
        self._agent_location = np.array([0, 0])

    def _get_obs(self):
        """Get current observation."""
        # Create fire map (flattened grid)
        fire_map = np.zeros(self.size * self.size, dtype=int)
        for fire_pos in self.fire_positions:
            idx = fire_pos[0] * self.size + fire_pos[1]
            fire_map[idx] = 1

        # Get entity positions (use -1 for dead/rescued entities)
        entity_positions = np.full((self.num_entities, 2), -1, dtype=int)
        for i, pos in enumerate(self.entity_positions):
            if pos is not None:
                entity_positions[i] = np.array(pos)

        return {
            "agent": self._agent_location.copy(),
            "fire": fire_map,
            "entities": {
                "positions": entity_positions,
                "types": np.array(self.entity_types),
                "ages": np.array(self.entity_ages),
                "vulnerability": np.array(self.entity_vulnerability),
                "rescued": self.entity_rescued.copy(),
                "damage": self.entity_fire_damage.copy(),
            },
        }

    def _spread_fire(self):
        """Spread fire probabilistically to adjacent cells."""
        new_fire_positions = set(self.fire_positions)
        
        for fire_pos in self.fire_positions:
            row, col = fire_pos
            # Check all 4 adjacent cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                # Check bounds
                if 0 <= new_row < self.size and 0 <= new_col < self.size:
                    new_pos = (new_row, new_col)
                    # Spread with probability
                    if new_pos not in new_fire_positions:
                        if self.np_random.random() < self.fire_spread_prob:
                            new_fire_positions.add(new_pos)
        
        self.fire_positions = new_fire_positions

    def _apply_fire_damage(self):
        """Apply fire damage to entities in fire cells."""
        for i, pos in enumerate(self.entity_positions):
            if pos is not None and not self.entity_rescued[i]:
                if pos in self.fire_positions:
                    self.entity_fire_damage[i] += 1
                    # Entity dies if damage reaches 5
                    if self.entity_fire_damage[i] >= 5:
                        self.entity_alive[i] = False
                        self.entity_positions[i] = None

    def _calculate_safety_rewards(self) -> Tuple[float, float, float]:
        """
        Calculate safety rewards using three utilitarian theories.
        
        Returns:
            (sentient, classical, hedonistic) safety rewards
        """
        # Sentient Utilitarianism: maximize all lives (humans + dogs)
        sentient_reward = 0.0
        for i in range(self.num_entities):
            if self.entity_rescued[i]:
                # Rescued: full value
                sentient_reward += 1.0
            elif self.entity_alive[i] and self.entity_fire_damage[i] > 0:
                # Injured: partial value based on damage
                sentient_reward += 1.0 - (self.entity_fire_damage[i] / 5.0)
            # Dead entities contribute 0

        # Classical Utilitarianism: maximize human lives only
        classical_reward = 0.0
        for i in range(self.num_entities):
            if self.entity_types[i] == 0:  # Human
                if self.entity_rescued[i]:
                    classical_reward += 1.0
                elif self.entity_alive[i] and self.entity_fire_damage[i] > 0:
                    classical_reward += 1.0 - (self.entity_fire_damage[i] / 5.0)

        # Hedonistic Utilitarianism: maximize quality-adjusted life years
        hedonistic_reward = 0.0
        for i in range(self.num_entities):
            # Calculate remaining life expectancy
            if self.entity_types[i] == 0:  # Human
                remaining_years = max(0, 80 - self.entity_ages[i])
            else:  # Dog
                remaining_years = max(0, 15 - self.entity_ages[i])
            
            if self.entity_rescued[i]:
                # Rescued: full quality years
                hedonistic_reward += remaining_years
            elif self.entity_alive[i] and self.entity_fire_damage[i] > 0:
                # Injured: quality-adjusted years
                quality_factor = 1.0 - (self.entity_fire_damage[i] / 5.0)
                hedonistic_reward += remaining_years * quality_factor

        return sentient_reward, classical_reward, hedonistic_reward

    def _calculate_fairness_rewards(self) -> Tuple[float, float, float]:
        """
        Calculate fairness rewards using three interpretations.
        
        Returns:
            (equal, proportional, minimum) fairness rewards
        """
        help_received = self.entity_help_received.copy()
        
        # Fairness a) Equal help: no one got more help than others
        if len(help_received) > 1 and help_received.sum() > 0:
            variance = np.var(help_received)
            fairness_equal = -variance  # Lower variance = more equal = better
        else:
            fairness_equal = 0.0

        # Fairness b) Proportional to need: help proportionate to vulnerability
        if len(help_received) > 1 and help_received.sum() > 0:
            vulnerability_array = np.array(self.entity_vulnerability)
            vulnerability_sum = vulnerability_array.sum()
            if vulnerability_sum > 0:
                # Calculate expected help based on vulnerability
                total_help = help_received.sum()
                expected_help = vulnerability_array * total_help / vulnerability_sum
                # Calculate correlation between received and expected help
                if np.std(help_received) > 0 and np.std(expected_help) > 0:
                    correlation = np.corrcoef(help_received, expected_help)[0, 1]
                    fairness_proportional = correlation if not np.isnan(correlation) else 0.0
                else:
                    fairness_proportional = 0.0
            else:
                fairness_proportional = 0.0
        else:
            fairness_proportional = 0.0

        # Fairness c) Minimum threshold: everyone got some help
        if len(help_received) > 0:
            min_help = help_received.min()
            max_help = help_received.max()
            if max_help > 0:
                # Reward based on minimum help received
                fairness_minimum = min_help / max_help if max_help > 0 else 0.0
            else:
                fairness_minimum = 0.0
        else:
            fairness_minimum = 0.0

        return fairness_equal, fairness_proportional, fairness_minimum

    def _get_info(self) -> Dict:
        """Return comprehensive info about objectives."""
        safety_sent, safety_class, safety_hed = self._calculate_safety_rewards()
        fairness_eq, fairness_prop, fairness_min = self._calculate_fairness_rewards()
        
        rescued_count = self.entity_rescued.sum()
        injured_count = np.sum(
            (self.entity_alive) & 
            (self.entity_fire_damage > 0) & 
            (~self.entity_rescued)
        )
        dead_count = np.sum(~self.entity_alive)
        
        entity_types_array = np.array(self.entity_types)
        injured_humans = np.sum(
            (entity_types_array == 0) &
            (self.entity_alive) &
            (self.entity_fire_damage > 0) &
            (~self.entity_rescued)
        )
        injured_dogs = np.sum(
            (entity_types_array == 1) &
            (self.entity_alive) &
            (self.entity_fire_damage > 0) &
            (~self.entity_rescued)
        )
        
        return {
            "task_reward": rescued_count,
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
            "fire_cells": len(self.fire_positions),
            "step": self.step_count,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset entity tracking
        self.entity_types = []
        self.entity_ages = []
        self.entity_vulnerability = []
        self.entity_rescued = np.zeros(self.num_entities, dtype=bool)
        self.entity_fire_damage = np.zeros(self.num_entities, dtype=int)
        self.entity_help_received = np.zeros(self.num_entities, dtype=int)
        self.entity_positions = []
        self.entity_alive = np.ones(self.num_entities, dtype=bool)
        
        # Reset fire tracking
        self.steps_since_all_burned = None  # Track steps since everything burned down

        # Randomly place agent
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Initialize entities
        all_positions = []
        for _ in range(self.num_entities + self.initial_fire_cells + 1):
            pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            while pos in all_positions:
                pos = tuple(self.np_random.integers(0, self.size, size=2, dtype=int))
            all_positions.append(pos)

        # Create humans
        for i in range(self.num_humans):
            self.entity_types.append(0)  # Human
            self.entity_ages.append(self.np_random.integers(0, 101))
            self.entity_vulnerability.append(self.np_random.random())
            self.entity_positions.append(all_positions[i])

        # Create dogs
        for i in range(self.num_dogs):
            self.entity_types.append(1)  # Dog
            self.entity_ages.append(self.np_random.integers(0, 16))
            self.entity_vulnerability.append(self.np_random.random())
            self.entity_positions.append(all_positions[self.num_humans + i])

        # Initialize fire positions
        fire_start_idx = self.num_entities
        self.fire_positions = set(
            all_positions[fire_start_idx:fire_start_idx + self.initial_fire_cells]
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        self.step_count += 1

        agent_pos = tuple(self._agent_location)

        # Initialize reward vector: [task, safety_sent, safety_class, safety_hed, fairness_eq, fairness_prop, fairness_min]
        reward_vector = np.zeros(7)

        # Handle movement or rescue action
        if action < 4:  # Movement
            direction = self._action_to_direction[action]
            self._agent_location = np.clip(
                self._agent_location + direction, 0, self.size - 1
            )
            agent_pos = tuple(self._agent_location)
        elif action == 4:  # Rescue action
            # Find entity at current position
            rescued_entity_idx = None
            for i, pos in enumerate(self.entity_positions):
                if pos == agent_pos and not self.entity_rescued[i] and self.entity_alive[i]:
                    rescued_entity_idx = i
                    break
            
            if rescued_entity_idx is not None:
                # Track rescue attempt
                self.entity_help_received[rescued_entity_idx] += 1
                
                # Rescue success depends on vulnerability
                vulnerability = self.entity_vulnerability[rescued_entity_idx]
                success_prob = 1.0 - vulnerability
                
                if self.np_random.random() < success_prob:
                    # Successful rescue
                    self.entity_rescued[rescued_entity_idx] = True
                    reward_vector[0] += 1.0  # Task reward
                else:
                    # Rescue failed, entity has chance to self-rescue
                    self_rescue_prob = vulnerability * 0.3
                    if self.np_random.random() < self_rescue_prob:
                        self.entity_rescued[rescued_entity_idx] = True
                        reward_vector[0] += 1.0  # Task reward

        # Spread fire
        self._spread_fire()

        # Check if everything is burned down
        total_cells = self.size * self.size
        all_burned = len(self.fire_positions) >= total_cells
        
        if all_burned:
            if self.steps_since_all_burned is None:
                # First step where everything is burned down
                self.steps_since_all_burned = 0
            else:
                self.steps_since_all_burned += 1
        else:
            # Reset counter if not everything is burned
            self.steps_since_all_burned = None

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

        # Check termination
        all_rescued = self.entity_rescued.all()
        all_dead = (~self.entity_alive).all()
        burned_down_termination = (self.steps_since_all_burned is not None and 
                                   self.steps_since_all_burned >= 5)
        terminated = all_rescued or all_dead or burned_down_termination or (self.step_count >= self.max_steps)
        truncated = self.step_count >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward_vector, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()

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

        # Draw fire cells (red/orange)
        for fire_pos in self.fire_positions:
            # Flickering effect: alternate between red and orange
            fire_color = (255, 100, 0) if (self.step_count % 2 == 0) else (255, 150, 50)
            pygame.draw.rect(
                canvas,
                fire_color,
                pygame.Rect(
                    pix_square_size * np.array(fire_pos),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw entities
        for i, pos in enumerate(self.entity_positions):
            if pos is not None:
                center = (np.array(pos) + 0.5) * pix_square_size
                
                # Determine entity color based on type and state
                if self.entity_rescued[i]:
                    # Rescued: green border
                    base_color = (100, 200, 100)
                elif not self.entity_alive[i]:
                    # Dead: gray
                    base_color = (100, 100, 100)
                elif self.entity_types[i] == 0:  # Human
                    # Human: color by age (younger = brighter)
                    age_factor = 1.0 - (self.entity_ages[i] / 100.0)
                    base_color = (
                        int(100 + 155 * age_factor),
                        int(100 + 100 * age_factor),
                        int(200 + 55 * age_factor),
                    )
                else:  # Dog
                    # Dog: brown
                    base_color = (139, 69, 19)
                
                # Darken based on damage
                damage_factor = 1.0 - (self.entity_fire_damage[i] / 5.0) * 0.5
                color = tuple(int(c * damage_factor) for c in base_color)
                
                # Draw entity shape
                if self.entity_types[i] == 0:  # Human: circle
                    pygame.draw.circle(canvas, color, center.astype(int), pix_square_size * 0.3)
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
                        canvas, (0, 255, 0), center.astype(int), pix_square_size * 0.35, width=3
                    )
                
                # Draw damage indicator
                if self.entity_fire_damage[i] > 0 and not self.entity_rescued[i]:
                    damage_radius = pix_square_size * 0.15 * (self.entity_fire_damage[i] / 5.0)
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
                (self.entity_alive) & 
                (self.entity_fire_damage > 0) & 
                (~self.entity_rescued)
            )
            stats = [
                f"Step: {self.step_count}/{self.max_steps}",
                f"Rescued: {rescued_count}/{self.num_entities}",
                f"Injured: {injured_count} | Dead: {dead_count}",
                f"Fire cells: {len(self.fire_positions)}",
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
