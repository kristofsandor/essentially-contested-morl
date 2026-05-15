"""Self-contained implementation of lexicographic RL agents.

Ported from https://github.com/lrhammond/lmorl/blob/main/src/agents/learners_lexicographic.py
with supporting network utilities from
https://github.com/lrhammond/lmorl/blob/main/src/agents/networks.py

Three algorithms:
  - LexDQN          : off-policy DQN with lexicographic action selection
  - LexActorCritic  : on-policy A2C/PPO with Lagrangian lexicographic constraints
  - LexTabular      : tabular Q-learning with lexicographic action selection

All accept a reward vector of shape (reward_size,) and treat objectives
in order of priority: index 0 is highest priority, index 1 is second, etc.
"""

import collections
import math
import pickle
import random
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------

@dataclass
class LexTrainParams:
    """Hyperparameters shared across lexicographic agent types."""

    # Number of reward / objective dimensions
    reward_size: int = 2

    # Exploration probability (epsilon-greedy)
    epsilon: float = 0.1

    # Replay buffer capacity
    buffer_size: int = 50_000

    # Mini-batch size
    batch_size: int = 64

    # Disable CUDA even if available
    no_cuda: bool = False

    # How many env steps between gradient updates (LexDQN)
    update_every: int = 4

    # How many gradient steps per update event (LexDQN)
    update_steps: int = 1

    # Lexicographic slack: an action is permissible for objective i if its
    # Q-value is within `slack * |max_Q|` of the optimal Q-value for objective i.
    slack: float = 0.01

    # Neural-network architecture: 'DNN' (fully-connected) or 'CNN'
    network: str = "DNN"

    # Adam learning rate
    learning_rate: float = 1e-3

    # LexTabular: True → SARSA (on-policy), False → Q-learning (off-policy)
    lextab_on_policy: bool = False


# ---------------------------------------------------------------------------
# Neural network modules
# ---------------------------------------------------------------------------

class DNN(nn.Module):
    """Two-layer fully-connected network."""

    def __init__(self, in_size: int, out_size, hidden: int = 64):
        super().__init__()
        self.out_size = out_size
        flat_out = int(np.prod(out_size))
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, flat_out)
        self.fc1.to(device)
        self.fc2.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if not isinstance(self.out_size, int):
            x = x.view((x.size(0),) + tuple(self.out_size))
        return x


class PolicyDNN(nn.Module):
    """Two-layer fully-connected policy network (outputs softmax probabilities)."""

    def __init__(self, in_size: int, action_size: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden)
        self.fc2 = nn.Linear(hidden, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1).float()
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    Experience = collections.namedtuple(
        "Experience",
        ["state", "action", "reward", "next_state", "done"],
    )

    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.Experience(state, action, reward, next_state, done))

    def sample(self, sample_all: bool = False):
        batch = list(self.memory) if sample_all else random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state.cpu().numpy() for e in batch])
        ).float().to(device)

        # actions may be int or 1-D tensor
        raw_actions = []
        for e in batch:
            a = e.action
            if isinstance(a, torch.Tensor):
                raw_actions.append(a.cpu().numpy().reshape(1))
            else:
                raw_actions.append(np.array([a]))
        actions = torch.from_numpy(np.vstack(raw_actions)).long().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in batch])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state.cpu().numpy() for e in batch])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([float(e.done) for e in batch])
        ).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# LexDQN
# ---------------------------------------------------------------------------

class LexDQN:
    """Lexicographic Deep Q-Network (off-policy).

    Maintains a single multi-headed Q-network that outputs
    Q-values for every (objective, action) pair. At each decision
    step, it iterates through objectives in priority order and prunes
    the set of permissible actions using the slack parameter.
    """

    def __init__(
        self,
        train_params: LexTrainParams,
        in_size: int,
        action_size: int,
        hidden: int = 64,
    ):
        self.epsilon = train_params.epsilon
        self.buffer_size = train_params.buffer_size
        self.batch_size = train_params.batch_size
        self.update_every = train_params.update_every
        self.update_steps = train_params.update_steps
        self.slack = train_params.slack
        self.reward_size = train_params.reward_size
        self.discount = 0.99

        self.actions = list(range(action_size))
        self.action_size = action_size
        self.t = 0

        # Multi-headed Q-network: output shape (reward_size, action_size)
        self.model = DNN(in_size, (self.reward_size, action_size), hidden)
        if torch.cuda.is_available() and not train_params.no_cuda:
            self.model.cuda()
        self.model.eval()

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=train_params.learning_rate)

    # ------------------------------------------------------------------ #

    def act(self, state: torch.Tensor) -> int:
        """Epsilon-greedy with lexicographic tie-breaking."""
        if np.random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            Q = self.model(state)[0]  # (reward_size, action_size)
        return random.choice(self._permissible_actions(Q))

    def _permissible_actions(self, Q: torch.Tensor) -> List[int]:
        """Return actions that are not pruned by any higher-priority objective."""
        perm = list(self.actions)
        for i in range(self.reward_size):
            Qi = Q[i, :]
            m = max(Qi[a].item() for a in perm)
            r = self.slack
            perm = [a for a in perm if Qi[a].item() >= m - r * abs(m)]
        return perm

    def _lexmax(self, Q: torch.Tensor) -> torch.Tensor:
        """Return the Q-value vector for the lexicographically optimal action."""
        a = self._permissible_actions(Q)[0]
        return Q[:, a]

    # ------------------------------------------------------------------ #

    def step(self, state, action, reward, next_state, done):
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)
        if self.t % self.update_every == 0 and len(self.memory) > self.batch_size:
            for _ in range(self.update_steps):
                self._update(self.memory.sample())

    def _update(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        self.model.train()

        # actions: (batch, 1) → expand to (batch, reward_size, 1)
        idx = actions.unsqueeze(1).expand(-1, self.reward_size, 1)
        predictions = self.model(states).gather(2, idx).squeeze(2)  # (batch, reward_size)

        with torch.no_grad():
            Q_next = self.model(next_states)  # (batch, reward_size, action_size)
            next_vals = torch.stack(
                [self._lexmax(Q) for Q in torch.unbind(Q_next, dim=0)], dim=0
            )  # (batch, reward_size)

        targets = rewards + self.discount * next_vals * (1 - dones)
        loss = nn.MSELoss()(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    # ------------------------------------------------------------------ #

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), f"{path}-lex_dqn.pt")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(f"{path}-lex_dqn.pt"))


# ---------------------------------------------------------------------------
# LexActorCritic
# ---------------------------------------------------------------------------

class LexActorCritic:
    """Lexicographic Actor-Critic with Lagrangian soft constraints.

    Supports both A2C (``mode='a2c'``) and PPO (``mode='ppo'``) objectives.
    With ``sequential=False`` (default) all objectives are updated together
    using Lagrange multipliers that enforce lexicographic priority.
    """

    def __init__(
        self,
        train_params: LexTrainParams,
        in_size: int,
        action_size: int,
        mode: str = "a2c",
        second_order: bool = False,
        sequential: bool = False,
        hidden: int = 64,
    ):
        assert mode in ("a2c", "ppo"), "mode must be 'a2c' or 'ppo'"

        self.reward_size = train_params.reward_size
        self.batch_size = train_params.batch_size
        self.buffer_size = train_params.buffer_size
        self.mode = mode
        self.second_order = second_order  # second-order not supported here
        self.discount = 0.99
        self.t = 0

        self.actor = PolicyDNN(in_size, action_size, hidden)
        self.critic = DNN(in_size, self.reward_size, hidden)

        if torch.cuda.is_available() and not train_params.no_cuda:
            self.actor.cuda()
            self.critic.cuda()
        self.actor.eval()
        self.critic.eval()

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=train_params.learning_rate * 0.01
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=train_params.learning_rate
        )

        # Lagrange multipliers: one per lower-priority objective
        self.mu: List[float] = [0.0] * (self.reward_size - 1)
        self.j: List[float] = [0.0] * (self.reward_size - 1)
        self.recent_losses = [
            collections.deque(maxlen=50) for _ in range(self.reward_size)
        ]

        # beta[i] is the base weight for objective i (highest priority gets
        # highest beta via reversed ordering)
        self.beta = list(reversed(range(1, self.reward_size + 1)))
        self.eta = [1e-3 * b for b in self.beta]

        # For sequential updating: track current objective index
        self.i = 0 if sequential else None

        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        if mode == "ppo":
            self.kl_weight = 1.0
            self.kl_target = 0.025

    # ------------------------------------------------------------------ #

    def act(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            probs = self.actor(state)
        return Categorical(probs).sample()

    # ------------------------------------------------------------------ #

    def step(self, state, action, reward, next_state, done):
        if self.t == 0:
            self.start_state = state
        self.t += 1
        self.memory.add(state, action, reward, next_state, done)
        if self.t % self.batch_size == 0:
            self._update(self.memory.sample(sample_all=True))
            self.memory.memory.clear()

    # ------------------------------------------------------------------ #

    def _get_log_probs(self, states, actions):
        dists = self.actor(states)
        log_probs = torch.log(torch.gather(dists, 1, actions) + 1e-8)
        return log_probs.squeeze(1)  # (batch,)

    def _compute_loss(self, experiences, reward_range: int):
        states, actions, rewards, next_states, dones = experiences

        # First-order weights incorporating Lagrange multipliers
        first_order = []
        for i in range(reward_range - 1):
            if self.i is not None:
                w = self.beta[reward_range - 1] * self.mu[i]
            else:
                w = self.beta[i] + self.mu[i] * sum(
                    self.beta[j] for j in range(i + 1, reward_range)
                )
            first_order.append(w)
        first_order.append(self.beta[reward_range - 1])
        first_order_weights = torch.tensor(first_order, dtype=torch.float32)

        if self.mode == "a2c":
            with torch.no_grad():
                baseline = self.critic(states)
                outcome = rewards + self.discount * self.critic(next_states) * (1 - dones)
                advantage = (outcome - baseline).detach()

            log_probs = self._get_log_probs(states, actions)
            weighted_adv = torch.sum(
                first_order_weights * advantage[:, :reward_range], dim=1
            )
            loss = -(log_probs * weighted_adv).mean()

            for i in range(self.reward_size):
                self.recent_losses[i].append(
                    -(log_probs * advantage[:, i]).mean().detach()
                )

        else:  # ppo
            with torch.no_grad():
                baseline = self.critic(states)
                outcome = rewards + self.discount * self.critic(next_states) * (1 - dones)
                advantage = (outcome - baseline).detach()
                old_log_probs = self._get_log_probs(states, actions).detach()

            new_log_probs = self._get_log_probs(states, actions)
            ratios = torch.exp(new_log_probs - old_log_probs)
            kl_penalty = new_log_probs - old_log_probs
            weighted_adv = torch.sum(
                first_order_weights * advantage[:, :reward_range], dim=1
            )
            loss = -(ratios * weighted_adv - self.kl_weight * kl_penalty).mean()

            for i in range(self.reward_size):
                rel_kl_w = (
                    self.kl_weight * first_order_weights[i] / first_order_weights.sum()
                    if i < reward_range
                    else 0.0
                )
                self.recent_losses[i].append(
                    -(ratios * advantage[:, i] - rel_kl_w * kl_penalty).mean().detach()
                )

            if kl_penalty.mean() < self.kl_target / 1.5:
                self.kl_weight *= 0.5
            elif kl_penalty.mean() > self.kl_target * 1.5:
                self.kl_weight *= 2.0

        return loss

    def _update_actor(self, experiences):
        self.actor.train()
        reward_range = (
            self.i + 1 if self.i is not None else self.reward_size
        )
        loss = self._compute_loss(experiences, reward_range)
        if torch.isnan(loss) or torch.isinf(loss):
            self.actor.eval()
            return
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.actor.eval()

    def _update_critic(self, experiences):
        states, _, rewards, next_states, dones = experiences
        self.critic.train()
        pred = self.critic(states)
        with torch.no_grad():
            target = rewards + self.discount * self.critic(next_states) * (1 - dones)
        loss = nn.MSELoss()(pred, target)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        self.critic.eval()

    def _update_lagrange(self):
        """Update Lagrange multipliers to enforce lexicographic priority."""
        if self.i is not None:
            if not self._converged():
                if self.i != self.reward_size - 1:
                    losses = self.recent_losses[self.i]
                    if len(losses) >= 26:
                        self.j[self.i] = -torch.stack(list(losses)[25:]).mean().item()
            else:
                self.i = 0 if self.i == self.reward_size - 1 else self.i + 1
        else:
            for i in range(self.reward_size - 1):
                losses = self.recent_losses[i]
                if len(losses) >= 26:
                    self.j[i] = -torch.stack(list(losses)[25:]).mean().item()

        r = self.i if self.i is not None else self.reward_size - 1
        for i in range(r):
            losses = self.recent_losses[i]
            if len(losses) > 0:
                current_loss = -losses[-1].item()
                self.mu[i] += self.eta[i] * (self.j[i] - current_loss)
                self.mu[i] = max(self.mu[i], 0.0)

    def _converged(self, tolerance: float = 0.1, minimum_updates: int = 50) -> bool:
        if self.i is None:
            return False
        losses = self.recent_losses[self.i]
        if len(losses) < minimum_updates:
            return False
        l_old = torch.stack(list(losses)[:24]).mean().float()
        l_new = torch.stack(list(losses)[25:]).mean().float()
        if l_new.abs() < 1e-8:
            return True
        return (l_old - l_new).abs() / l_new.abs() <= tolerance

    def _update(self, experiences):
        self._update_actor(experiences)
        self._update_critic(experiences)
        self._update_lagrange()

    # ------------------------------------------------------------------ #

    def save_model(self, path: str):
        torch.save(self.actor.state_dict(), f"{path}-lex_ac_actor.pt")
        torch.save(self.critic.state_dict(), f"{path}-lex_ac_critic.pt")

    def load_model(self, path: str):
        self.actor.load_state_dict(torch.load(f"{path}-lex_ac_actor.pt"))
        self.critic.load_state_dict(torch.load(f"{path}-lex_ac_critic.pt"))


# ---------------------------------------------------------------------------
# LexTabular
# ---------------------------------------------------------------------------

class LexTabular:
    """Lexicographic tabular Q-learning / SARSA.

    States are stored in a dictionary keyed by string representation,
    making this suitable for small, discrete observation spaces.
    """

    def __init__(
        self,
        train_params: LexTrainParams,
        action_size: int,
        initialisation=0.0,
        double: bool = False,
    ):
        self.slack = train_params.slack
        self.epsilon = train_params.epsilon
        self.reward_size = train_params.reward_size
        self.discount = 0.99
        self.actions = list(range(action_size))
        self.double = double

        if isinstance(initialisation, (float, int)):
            self.initialisation = [float(initialisation)] * self.reward_size
        else:
            self.initialisation = list(initialisation)

        if not double:
            self.Q = [{} for _ in range(self.reward_size)]
        else:
            self.Qa = [{} for _ in range(self.reward_size)]
            self.Qb = [{} for _ in range(self.reward_size)]

        # Choose update rule
        if double:
            self._step_fn = self._double_Q_update
        elif train_params.lextab_on_policy:
            self._step_fn = self._sarsa_update
        else:
            self._step_fn = self._q_update

    # ------------------------------------------------------------------ #

    def act(self, state) -> int:
        s = str(state)
        self._init_state(s)
        return self._lex_epsilon_greedy(s)

    def step(self, state, action, reward, next_state, done):
        self._step_fn(state, action, reward, next_state, done)

    # ------------------------------------------------------------------ #

    def _init_state(self, state: str):
        if not self.double:
            if state not in self.Q[0]:
                for i, Qi in enumerate(self.Q):
                    Qi[state] = {a: self.initialisation[i] for a in self.actions}
        else:
            if state not in self.Qa[0]:
                for i, Qai in enumerate(self.Qa):
                    Qai[state] = {a: self.initialisation[i] for a in self.actions}
                for i, Qbi in enumerate(self.Qb):
                    Qbi[state] = {a: self.initialisation[i] for a in self.actions}

    def _lex_epsilon_greedy(self, state: str) -> int:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        perm = list(self.actions)
        if not self.double:
            for Qi in self.Q:
                m = max(Qi[state][a] for a in perm)
                r = self.slack
                perm = [a for a in perm if Qi[state][a] >= m - r * abs(m)]
        else:
            for Qai, Qbi in zip(self.Qa, self.Qb):
                m = max(0.5 * (Qai[state][a] + Qbi[state][a]) for a in perm)
                r = self.slack
                perm = [
                    a
                    for a in perm
                    if 0.5 * (Qai[state][a] + Qbi[state][a]) >= m - r * abs(m)
                ]
        return np.random.choice(perm)

    # ------------------------------------------------------------------ #

    def _q_update(self, state, action, reward, next_state, done):
        s = str(state)
        ns = str(next_state)
        self._init_state(s)
        self._init_state(ns)

        perm = list(self.actions)
        alpha = 0.01
        for i, Qi in enumerate(self.Q):
            m = max(Qi[ns][a] for a in perm)
            r = self.slack
            perm = [a for a in perm if Qi[ns][a] >= m - r * abs(m)]
            target = reward[i] + (0.0 if done else self.discount * m)
            Qi[s][action] = (1 - alpha) * Qi[s][action] + alpha * target

    def _sarsa_update(self, state, action, reward, next_state, done):
        s = str(state)
        ns = str(next_state)
        self._init_state(s)
        self._init_state(ns)

        # Compute permissible actions for next state
        perm = list(self.actions)
        for Qi in self.Q:
            m = max(Qi[ns][a] for a in perm)
            r = self.slack
            perm = [a for a in perm if Qi[ns][a] >= m - r * abs(m)]

        # On-policy expected value under epsilon-greedy policy
        alpha = 0.01
        eps = self.epsilon
        n_a = len(self.actions)
        n_p = len(perm)
        for i, Qi in enumerate(self.Q):
            exp = sum(
                (
                    (1 - eps) / n_p + eps / n_a
                    if a in perm
                    else eps / n_a
                )
                * Qi[ns][a]
                for a in self.actions
            )
            target = reward[i] + (0.0 if done else self.discount * exp)
            Qi[s][action] = (1 - alpha) * Qi[s][action] + alpha * target

    def _double_Q_update(self, state, action, reward, next_state, done):
        s = str(state)
        ns = str(next_state)
        self._init_state(s)
        self._init_state(ns)

        perm = list(self.actions)
        r = self.slack
        alpha = 0.01

        for i, (Qai, Qbi) in enumerate(zip(self.Qa, self.Qb)):
            if np.random.random() < 0.5:
                m = max(Qbi[ns][a] for a in perm)
                perm = [a for a in perm if Qbi[ns][a] >= m - r * abs(m)]
                a_star = np.random.choice(perm)
                m = Qai[ns][a_star]
                target = reward[i] + (0.0 if done else self.discount * m)
                Qai[s][action] = (1 - alpha) * Qai[s][action] + alpha * target
            else:
                m = max(Qai[ns][a] for a in perm)
                perm = [a for a in perm if Qai[ns][a] >= m - r * abs(m)]
                a_star = np.random.choice(perm)
                m = Qbi[ns][a_star]
                target = reward[i] + (0.0 if done else self.discount * m)
                Qbi[s][action] = (1 - alpha) * Qbi[s][action] + alpha * target

    # ------------------------------------------------------------------ #

    def save_model(self, path: str):
        if not self.double:
            with open(f"{path}-lex_tabular.pkl", "wb") as f:
                pickle.dump(self.Q, f)
        else:
            with open(f"{path}-lex_tabular_A.pkl", "wb") as f:
                pickle.dump(self.Qa, f)
            with open(f"{path}-lex_tabular_B.pkl", "wb") as f:
                pickle.dump(self.Qb, f)

    def load_model(self, path: str):
        if not self.double:
            with open(f"{path}-lex_tabular.pkl", "rb") as f:
                self.Q = pickle.load(f)
        else:
            with open(f"{path}-lex_tabular_A.pkl", "rb") as f:
                self.Qa = pickle.load(f)
            with open(f"{path}-lex_tabular_B.pkl", "rb") as f:
                self.Qb = pickle.load(f)
