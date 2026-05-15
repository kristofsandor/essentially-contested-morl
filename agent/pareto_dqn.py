"""Pareto-DQN (PDQN).

Approximates the Pareto front for multi-objective RL with high-dimensional
state-spaces by extending DQN with:
  - A reward network  R(s, a) → R^d  (stationary target, no target net)
  - A non-dominated-set network  NDt(s, a, o_{1..d-1}) → o_d
  - A target network for NDt + experience replay
  - Hypervolume-based ε-greedy action selection

Based on:
    M. Reymond and A. Nowé, "Pareto-DQN: Approximating the Pareto front in
    complex multi-objective decision problems," ALA 2019 Workshop.

Architecture (paper §3):
    The NDt network takes (state, action, o_1 ... o_{d-1}) as input and
    predicts o_d.  For the 2-objective case, o_1 is the single prefix and o_2
    is predicted.  The Pareto front is recovered by sampling many prefix values
    and predicting the corresponding last-objective value.

Sampling strategies (paper §3.2, Figs 1a–1c):
    • Out-of-range prefix samples get a target worse than any feasible reward
      for objective d (so the net learns to flag them as invalid).
    • Terminal transitions: the PF is a single point (the terminal reward).
      Dominated samples get the dominating point's value; non-dominated
      out-of-range samples get the reference-point value.
    • After applying Qset = R ⊕ γ·NDt the covered range shifts; uncovered
      regions are filled with the same dominated/reference strategy.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume


# ═══════════════════════════════════════════════════════════════════════════
#  Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Uniform experience-replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════
#  Neural Networks
# ═══════════════════════════════════════════════════════════════════════════

class RewardNet(nn.Module):
    """Estimates the expected immediate reward vector R(s, a) ∈ R^d.

    Architecture mirrors the paper: separate branches for state and action
    that are merged after one hidden layer.

    Input:  state_dim + num_actions  (action is one-hot)
    Output: reward_dim
    """

    def __init__(self, state_dim: int, num_actions: int, reward_dim: int,
                 hidden: int = 128):
        super().__init__()
        self.num_actions = num_actions
        # State branch
        self.fc_state = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        # Action branch
        self.fc_action = nn.Sequential(
            nn.Linear(num_actions, num_actions),
            nn.ReLU(),
        )
        # Merged
        self.fc_merge = nn.Sequential(
            nn.Linear(hidden + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, reward_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim)
            action: (B,) int64 action indices
        Returns:
            (B, reward_dim) estimated immediate reward
        """
        one_hot = torch.zeros(state.size(0), self.num_actions, device=state.device)
        one_hot.scatter_(1, action.unsqueeze(1), 1.0)
        hs = self.fc_state(state)
        ha = self.fc_action(one_hot)
        return self.fc_merge(torch.cat([hs, ha], dim=1))


class NDtNet(nn.Module):
    """Estimates NDt(s, a, o_1 ... o_{d-1}) → o_d.

    Paper architecture (§4.1):
        FC_s(state_dim + (d-1), H)  ──→  FC(H + num_actions, H) → FC(H, 1)
        FC_a(num_actions, num_actions)  ──↗

    The prefix objectives o_1 ... o_{d-1} are concatenated with the state.
    """

    def __init__(self, state_dim: int, num_actions: int, reward_dim: int,
                 hidden: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.reward_dim = reward_dim
        prefix_dim = reward_dim - 1  # d-1 objectives as input

        # State+prefix branch
        self.fc_state = nn.Sequential(
            nn.Linear(state_dim + prefix_dim, hidden),
            nn.ReLU(),
        )
        # Action branch
        self.fc_action = nn.Sequential(
            nn.Linear(num_actions, num_actions),
            nn.ReLU(),
        )
        # Merged → predict o_d
        self.fc_merge = nn.Sequential(
            nn.Linear(hidden + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                obj_prefix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:      (B, state_dim)
            action:     (B,) int64
            obj_prefix: (B, d-1) — first d-1 objective values
        Returns:
            (B, 1) predicted last objective value o_d
        """
        one_hot = torch.zeros(state.size(0), self.num_actions, device=state.device)
        one_hot.scatter_(1, action.unsqueeze(1), 1.0)
        hs = self.fc_state(torch.cat([state, obj_prefix], dim=1))
        ha = self.fc_action(one_hot)
        return self.fc_merge(torch.cat([hs, ha], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
#  Pareto-DQN
# ═══════════════════════════════════════════════════════════════════════════

class ParetoDQN:
    """Pareto-DQN (PDQN) algorithm.

    Args:
        state_dim:   Dimensionality of the (flat) state vector.
        num_actions: Number of discrete actions.
        reward_dim:  Number of objectives (d).
        ref_point:   Reference point for hypervolume (d,); must be strictly
                     dominated by all feasible Pareto-front points.
        reward_min:  Per-objective minimum feasible reward (d,).  Used to
                     assign "worse-than-min" targets to out-of-range samples.
        reward_max:  Per-objective maximum feasible reward (d,).  Together with
                     reward_min, defines the sampling range for prefixes.
        n_samples:   Number of prefix samples to approximate the PF.
        gamma:       Discount factor.
        lr_reward:   Learning rate for the reward network.
        lr_ndt:      Learning rate for the NDt network.
        buffer_size: Replay-buffer capacity.
        batch_size:  Mini-batch size for gradient updates.
        target_update_freq: Steps between target-network hard copies.
        initial_epsilon:    Starting ε for ε-greedy.
        final_epsilon:      Final ε.
        epsilon_decay_steps: Linear decay horizon.
        hidden:      Hidden-layer width.
        device:      PyTorch device string.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        reward_dim: int = 2,
        ref_point: Optional[np.ndarray] = None,
        reward_min: Optional[np.ndarray] = None,
        reward_max: Optional[np.ndarray] = None,
        n_samples: int = 32,
        gamma: float = 1.0,
        lr_reward: float = 1e-3,
        lr_ndt: float = 1e-4,
        buffer_size: int = 50_000,
        batch_size: int = 32,
        target_update_freq: int = 500,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reward_dim = reward_dim
        self.ref_point = np.array(ref_point) if ref_point is not None \
            else np.full(reward_dim, -1.0)
        self.reward_min = np.array(reward_min) if reward_min is not None \
            else np.full(reward_dim, -20.0)
        self.reward_max = np.array(reward_max) if reward_max is not None \
            else np.full(reward_dim, 130.0)
        self.n_samples = n_samples
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = initial_epsilon
        self.device = torch.device(device)

        # Worse-than-reference value for the last objective (paper §3.2)
        self.worse_value = self.ref_point[-1] - 1.0

        # Networks ----------------------------------------------------------
        self.reward_net = RewardNet(
            state_dim, num_actions, reward_dim, hidden
        ).to(self.device)
        self.ndt_net = NDtNet(
            state_dim, num_actions, reward_dim, hidden
        ).to(self.device)
        self.ndt_target = NDtNet(
            state_dim, num_actions, reward_dim, hidden
        ).to(self.device)
        self.ndt_target.load_state_dict(self.ndt_net.state_dict())
        self.ndt_target.eval()

        self.opt_reward = optim.Adam(self.reward_net.parameters(), lr=lr_reward)
        self.opt_ndt = optim.Adam(self.ndt_net.parameters(), lr=lr_ndt)

        # Replay buffer & counters -----------------------------------------
        self.buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    # ── prefix sampling ──────────────────────────────────────────────────

    def _sample_prefixes(self, n: int) -> np.ndarray:
        """Sample n prefix vectors from R^{d-1}.

        For d=2 this is a 1-D uniform sample over [reward_min[0], reward_max[0]].
        For d>2 it would be a (n, d-1) matrix.
        """
        lows = self.reward_min[: self.reward_dim - 1]
        highs = self.reward_max[: self.reward_dim - 1]
        return np.random.uniform(lows, highs, size=(n, self.reward_dim - 1)).astype(
            np.float32
        )

    # ── Pareto-front prediction ───────────────────────────────────────────

    @torch.no_grad()
    def _predict_ndt_points(
        self,
        state: np.ndarray,
        action: int,
        net: NDtNet,
        prefixes: np.ndarray,
    ) -> np.ndarray:
        """Query the NDt network for a batch of prefix samples.

        Args:
            state:    (state_dim,) numpy.
            action:   int.
            net:      NDtNet (online or target).
            prefixes: (n, d-1) numpy prefix values.

        Returns:
            (n, d) numpy points in NDt space (prefix ‖ predicted_od).
        """
        n = len(prefixes)
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        s_t = s_t.unsqueeze(0).expand(n, -1)
        a_t = torch.full((n,), action, dtype=torch.long, device=self.device)
        p_t = torch.tensor(prefixes, dtype=torch.float32, device=self.device)

        od_pred = net(s_t, a_t, p_t).squeeze(-1).cpu().numpy()  # (n,)
        return np.concatenate([prefixes, od_pred[:, None]], axis=1)  # (n, d)

    def predict_qset(
        self,
        state: np.ndarray,
        action: int,
        net: Optional[NDtNet] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Qset(s, a, p) = R(s, a) + γ · NDt(s, a, p).

        Samples prefix points, predicts the PF, adds R, and returns the
        non-dominated subset (filtering out invalid predictions).

        Returns:
            (M, d) numpy array of non-dominated Q-vectors.
        """
        if net is None:
            net = self.ndt_net
        n = n_samples or self.n_samples

        # Immediate reward estimate
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = torch.tensor([action], dtype=torch.long, device=self.device)
        with torch.no_grad():
            r_hat = self.reward_net(s_t, a_t).squeeze(0).cpu().numpy()  # (d,)

        # Sample prefixes and predict NDt
        prefixes = self._sample_prefixes(n)
        ndt_pts = self._predict_ndt_points(state, action, net, prefixes)  # (n, d)

        # Qset = R + γ · NDt
        qset_pts = r_hat + self.gamma * ndt_pts  # (n, d)

        # Filter: keep only points whose last-objective is above worse_value
        valid = qset_pts[:, -1] > self.worse_value + 0.5
        qset_pts = qset_pts[valid]

        if len(qset_pts) == 0:
            return r_hat.reshape(1, -1)

        # Return non-dominated set (using morl_baselines utility)
        candidates = {tuple(p) for p in qset_pts}
        nd_set = get_non_dominated(candidates)
        return np.array(list(nd_set))

    # ── Action selection ──────────────────────────────────────────────────

    def score_action(self, state: np.ndarray, action: int) -> float:
        """Hypervolume of Qset(s, a) w.r.t. self.ref_point."""
        qset = self.predict_qset(state, action)
        return hypervolume(self.ref_point, list(qset))

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection based on per-action hypervolume."""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        scores = np.array([self.score_action(state, a)
                           for a in range(self.num_actions)])
        return int(np.random.choice(np.flatnonzero(scores == scores.max())))

    # ── Target construction (paper Algorithm 1) ───────────────────────────

    def _build_ndt_targets(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-sample NDt training targets.

        For each transition in the batch:
          - If terminal:  target PF is just the terminal reward.
          - If not:       target = ND( ∪_{a'} Qset_target(s', a', p) )

        Then for each prefix sample we assign an o_d target using the
        sampling strategy of Figs 1b/1c.

        Returns:
            s_all:      (total, state_dim) states
            a_all:      (total,) actions
            prefix_all: (total, d-1) prefix inputs
            target_all: (total,) o_d targets for NDt
        """
        B = len(states)
        s_list, a_list, p_list, y_list = [], [], [], []

        for i in range(B):
            prefixes = self._sample_prefixes(self.n_samples)  # (n, d-1)

            if dones[i]:
                # Terminal: PF is the single terminal reward
                terminal_r = rewards[i]  # (d,)
                target_od = self._assign_targets_around_points(
                    prefixes, terminal_r.reshape(1, -1)
                )
            else:
                # Non-terminal: collect union of target Qsets over actions
                union_pts = []
                for a in range(self.num_actions):
                    qset = self.predict_qset(
                        next_states[i], a, net=self.ndt_target,
                        n_samples=self.n_samples,
                    )
                    union_pts.append(qset)
                union_pts = np.concatenate(union_pts, axis=0)

                cands = {tuple(p) for p in union_pts}
                nd_set = get_non_dominated(cands)
                nd_pts = np.array(list(nd_set))

                target_od = self._assign_targets_around_points(
                    prefixes, nd_pts
                )

            s_list.append(np.tile(states[i], (self.n_samples, 1)))
            a_list.append(np.full(self.n_samples, actions[i], dtype=np.int64))
            p_list.append(prefixes)
            y_list.append(target_od)

        return (
            torch.tensor(np.concatenate(s_list), dtype=torch.float32, device=self.device),
            torch.tensor(np.concatenate(a_list), dtype=torch.long, device=self.device),
            torch.tensor(np.concatenate(p_list), dtype=torch.float32, device=self.device),
            torch.tensor(np.concatenate(y_list), dtype=torch.float32, device=self.device),
        )

    def _assign_targets_around_points(
        self,
        prefixes: np.ndarray,    # (n, d-1)
        pf_points: np.ndarray,   # (M, d) — target PF points
    ) -> np.ndarray:
        """Assign an o_d target for each prefix using the paper's sampling strategy.

        For each prefix sample:
          - Find the closest PF point in the prefix dimensions.
          - If close enough → use that point's o_d as target.
          - If not on the PF (out-of-range / dominated) → use worse_value.

        This implements the strategies from Figures 1a, 1b, 1c of the paper.
        """
        n = len(prefixes)
        targets = np.full(n, self.worse_value, dtype=np.float32)

        if len(pf_points) == 0:
            return targets

        pf_prefix = pf_points[:, :-1]  # (M, d-1)
        pf_od = pf_points[:, -1]       # (M,)

        # Distance from each sample prefix to each PF prefix  →  (n, M)
        dists = np.linalg.norm(
            prefixes[:, None, :] - pf_prefix[None, :, :], axis=2
        )
        closest_idx = np.argmin(dists, axis=1)  # (n,)
        closest_dist = dists[np.arange(n), closest_idx]

        # Adaptive threshold: scale with the PF range / n_samples
        prefix_range = np.linalg.norm(self.reward_max[:-1] - self.reward_min[:-1])
        threshold = prefix_range / max(self.n_samples, 1) * 3.0

        # Close enough → PF value  (Fig 1a)
        close_mask = closest_dist < threshold
        targets[close_mask] = pf_od[closest_idx[close_mask]]

        # Out-of-range but dominated by a PF point → dominating point's o_d  (Figs 1b/1c)
        for k in range(n):
            if not close_mask[k]:
                dominated_by = np.all(pf_prefix >= prefixes[k], axis=1)
                if np.any(dominated_by):
                    dom_dists = dists[k].copy()
                    dom_dists[~dominated_by] = np.inf
                    targets[k] = pf_od[np.argmin(dom_dists)]

        return targets

    # ── Gradient step ─────────────────────────────────────────────────────

    def train_step(self) -> Optional[Dict[str, float]]:
        """Sample a mini-batch and perform one gradient update.

        Returns dict of losses or None if buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        s_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # ── Reward network (stationary target — no target net needed) ──
        r_pred = self.reward_net(s_t, a_t)
        loss_r = nn.functional.mse_loss(r_pred, r_t)
        self.opt_reward.zero_grad()
        loss_r.backward()
        self.opt_reward.step()

        # ── NDt network (Algorithm 1) ─────────────────────────────────
        s_all, a_all, p_all, y_all = self._build_ndt_targets(
            states, actions, rewards, next_states, dones
        )

        if len(s_all) > 0:
            # Predict Qset_od(s, a, p) = R(s,a)[-1] + γ · NDt(s, a, prefix)
            with torch.no_grad():
                r_hat = self.reward_net(s_all, a_all)  # (total, d)
            ndt_pred = self.ndt_net(s_all, a_all, p_all)  # (total, 1)
            qset_od_pred = r_hat[:, -1] + self.gamma * ndt_pred.squeeze(-1)

            loss_ndt = nn.functional.mse_loss(qset_od_pred, y_all)
            self.opt_ndt.zero_grad()
            loss_ndt.backward()
            self.opt_ndt.step()
        else:
            loss_ndt = torch.tensor(0.0)

        # ── Target network copy (every C steps) ─────────────────────
        if self.global_step % self.target_update_freq == 0:
            self.ndt_target.load_state_dict(self.ndt_net.state_dict())

        return {"loss_reward": loss_r.item(), "loss_ndt": loss_ndt.item()}

    # ── High-level API ────────────────────────────────────────────────────

    def _update_epsilon(self):
        progress = min(self.global_step / max(self.epsilon_decay_steps, 1), 1.0)
        self.epsilon = self.initial_epsilon + progress * (
            self.final_epsilon - self.initial_epsilon
        )

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Store transition, update networks, advance step counter."""
        self.buffer.push(state, action, reward, next_state, done)
        self.global_step += 1
        self._update_epsilon()
        return self.train_step()

    def get_pareto_front(self, state: np.ndarray) -> np.ndarray:
        """Return the estimated Pareto front at *state* as (M, d) array.

        Computed as ND( ∪_a Qset(s, a) ).
        """
        all_pts = []
        for a in range(self.num_actions):
            all_pts.append(self.predict_qset(state, a))
        pts = np.concatenate(all_pts, axis=0)
        cands = {tuple(p) for p in pts}
        nd = get_non_dominated(cands)
        return np.array(sorted(nd, key=lambda x: x[0]))
