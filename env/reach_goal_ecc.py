from enum import IntEnum
import gymnasium as gym
from numpy import float32
import numpy as np
import pygame

RENDER_FPS = 5

# Normalizes utilitarian per-human reward so its expected value matches the
# deontological per-human reward. Ages ~ U(0.001, 1.0) ⇒ E[age] ≈ 0.5,
# so multiplying by 2 puts both interpretations on the same average scale.
UTIL_SCALE = 2.0


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ECCReachGoalEnv(gym.Env):
    def __init__(
        self,
        grid_size=10,
        num_humans=20,
        step_penalty=0.4,
        terminal_reward=2,
        help_reward=1,
        proximity_reward=0.05,
        obs_as_grid=True,
        render_mode=None,
    ):
        self.grid_size = grid_size
        self.num_humans = num_humans
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.obs_as_grid = obs_as_grid
        self.proximity_reward = proximity_reward
        self.action_space = gym.spaces.Discrete(len(Action))
        if obs_as_grid:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(3, self.grid_size, self.grid_size), dtype=float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=1, shape=(4 + self.num_humans * 4,), dtype=float32
            )

        # Reward is a flattened per-interpretation matrix with one row per ethical
        # interpretation, emitted row-major as a single vector:
        #   row 0 (deontological): [task, deont_help]
        #   row 1 (utilitarian):   [task, util_help]
        # -> [task, deont_help, task, util_help]
        # Column 0 (task) is shared across interpretations. ECCEnvelope reshapes
        # this back into rows and trains each network on a single row. Keeping the
        # reward flat lets MORecordEpisodeStatistics / the replay buffer treat it
        # as an ordinary vector reward.
        self.reward_space = gym.spaces.Box(
            low=np.array(
                [
                    [-step_penalty, -proximity_reward],
                    [-step_penalty, -proximity_reward * UTIL_SCALE],
                ],
                dtype=np.float32,
            ).flatten(),
            high=np.array(
                [
                    [terminal_reward, help_reward + proximity_reward],
                    [terminal_reward, (help_reward + proximity_reward) * UTIL_SCALE],
                ],
                dtype=np.float32,
            ).flatten(),
        )

        self.help_reward = help_reward
        (
            self.agent_pos,
            self.goal_pos,
            self.prev_pos,
            self.human_ages,
            self.human_positions,
            self.helped,
        ) = self.setup()
        self.render_mode = render_mode
        self.window = None
        self.reset()

    def setup(self):
        agent_pos = [0, 0]
        goal_pos = [0, self.grid_size - 1]  # top right
        prev_pos = [0, 0]
        human_ages = np.random.uniform(0.001, 1.0, self.num_humans).round(2)
        human_positions = self.distribute_humans_2(self.grid_size, self.num_humans)
        human_positions = np.array(
            sorted(human_positions, key=lambda pos: (pos[0], pos[1]))
        )  # sort by row, then column for consistency
        helped = np.zeros(len(human_positions), dtype=bool)
        return agent_pos, goal_pos, prev_pos, human_ages, human_positions, helped

    def reset(self, seed=None, options=None):
        (
            self.agent_pos,
            self.goal_pos,
            self.prev_pos,
            self.human_ages,
            self.human_positions,
            self.helped,
        ) = self.setup()

        if self.render_mode == "human":
            self.render()

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

    def distribute_humans_2(self, grid_size, num_humans):
        """Place humans across rows 1..grid_size-1 for graduated detour costs.

        Round-robin: 1 human → row 1; G-1 humans → one per row;
        2*(G-1) humans → two per row; etc. Columns randomized per row.
        Deep-first variant: doubles up on deep rows before shallow ones,
        keeping early humans cheap and later ones expensive.
        """
        n_rows = grid_size - 1  # exclude agent/goal row 0
        if num_humans > n_rows * grid_size:
            raise ValueError(
                f"num_humans={num_humans} exceeds capacity "
                f"{n_rows * grid_size} for grid_size={grid_size}"
            )

        row_assignments = [
            (n_rows - (i % n_rows)) if (i // n_rows) % 2 else (i % n_rows + 1)
            for i in range(num_humans)
        ]

        # Random distinct columns within each row
        from collections import Counter
        counts = Counter(row_assignments)
        positions = []
        for row, count in counts.items():
            cols = np.random.choice(grid_size, size=count, replace=False)
            for c in cols:
                positions.append([row, c])

        return np.array(sorted(positions, key=lambda p: (p[0], p[1])))

    def step(self, action):
        # Save the pre-movement position so shaping is computed against
        # the actual previous cell, not a stale [0, 0] from reset().
        self.prev_pos = list(self.agent_pos)
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
        # Reward matrix. Rows = interpretations, columns = objectives.
        #            task   help
        # deont [    v0  , v1_d   ]
        # util  [    v0  , v1_u   ]
        reward = np.zeros((2, 2), dtype=np.float32)
        reward[:, 0] -= self.step_penalty  # shared task penalty in every row

        # out of bounds, cannot move (agent stays in place but still pays the step cost)
        if y_pos < 0 or y_pos >= self.grid_size or x_pos < 0 or x_pos >= self.grid_size:
            return (
                self.get_obs(),
                reward.flatten(),
                terminated,
                False,
                {},
            )

        self.agent_pos = [x_pos, y_pos]

        # into a human cell -- credit both interpretations on first contact
        human_matches = np.where(
            (self.human_positions == self.agent_pos).all(axis=1)
        )[0]
        if len(human_matches) > 0:
            human_idx = human_matches[0]
            if not self.helped[human_idx]:
                self.helped[human_idx] = True
                # deontological: uniform reward per human
                reward[0, 1] += self.help_reward
                # utilitarian: reward proportional to age, normalized so E[per-human] matches deont
                reward[1, 1] += (
                    self.help_reward * self.human_ages[human_idx] * UTIL_SCALE
                )

        # into goal cell
        if self.agent_pos == self.goal_pos:
            reward[:, 0] += self.terminal_reward  # shared task reward
            terminated = True

        if self.render_mode == "human":
            self.render()

        # Reward shaping: dense signal toward unhelped humans.
        # Computed for both interpretations using their respective notions of "value".
        unhelped_mask = ~self.helped
        moved = self.prev_pos != self.agent_pos
        if unhelped_mask.any() and moved:
            unhelped_positions = self.human_positions[unhelped_mask]
            unhelped_ages = self.human_ages[unhelped_mask]
            prev_dists = np.abs(unhelped_positions - self.prev_pos).sum(axis=1)
            curr_dists = np.abs(unhelped_positions - self.agent_pos).sum(axis=1)

            # Deontological shaping: move toward the NEAREST unhelped human (age-agnostic).
            reward[0, 1] += self.proximity_reward * (
                prev_dists.min() - curr_dists.min()
            )

            # Utilitarian shaping: move toward the human with the best age/distance ratio
            # (greedy value-per-step heuristic). Use max(dists, 1) to avoid div-by-zero.
            prev_score = (
                unhelped_ages * UTIL_SCALE / np.maximum(prev_dists, 1)
            ).max()
            curr_score = (
                unhelped_ages * UTIL_SCALE / np.maximum(curr_dists, 1)
            ).max()
            reward[1, 1] += self.proximity_reward * (curr_score - prev_score)

        return (
            self.get_obs(),
            reward.flatten(),
            terminated,
            False,
            {},
        )

    def render(self):
        BLUE = (0, 0, 255)
        RED = (255, 0, 0)
        GREEN = (0, 128, 0)
        BLACK = (0, 0, 0)

        window_size = 455
        pix_square = window_size // self.grid_size

        # init pygame window
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((window_size, window_size))
            pygame.display.set_caption("ECC Reach Goal")
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_size, window_size))
        canvas.fill((255, 255, 255))

        pygame.font.init()
        self.font = pygame.font.SysFont(None, 48)
        img = self.font.render("G", True, GREEN)
        canvas.blit(img, (np.array(self.goal_pos) + 0.15) * pix_square)

        # draw agent
        pygame.draw.circle(
            canvas, BLUE, (np.array(self.agent_pos) + 0.5) * pix_square, pix_square // 3
        )

        # draw unhelped humans
        for idx, (pos, age, helped) in enumerate(
            zip(self.human_positions, self.human_ages, self.helped)
        ):
            if not helped:
                color = (0, int(255 * age), 0)
                img = self.font.render("H", True, color)
                canvas.blit(img, (pos + 0.15) * pix_square)
            else:
                img = self.font.render("X", True, RED)
                canvas.blit(img, (pos + 0.15) * pix_square)

        self.window.blit(canvas, canvas.get_rect())
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Stopping render")
                self.close()
                self.render_mode = None
                return
        pygame.display.update()
        self.clock.tick(RENDER_FPS)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()