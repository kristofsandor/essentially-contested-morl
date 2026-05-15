"""Essentially Contested Concept Pareto-DQN (ECC-PDQN).

Extends Pareto-DQN to handle Essentially Contested Concepts (ECCs),
analogously to how ECC-PQL extends PQL.

Design (mirrors ECC-PQL):
    Each contested concept (e.g. Safety, Fairness) gets its own independent
    pair of (RewardNet, NDtNet + target) networks, each seeing only its
    concept-specific slice of the full reward vector.

    During action selection the per-concept Pareto fronts are scored
    (hypervolume or EUM) and combined across concepts — the product of
    per-concept scores encourages balanced exploration, exactly as in
    ECC-PQL's `score_hypervolume`.

    The *value-level Pareto front* lives in the space
    (concept_0_score, concept_1_score, ...) and captures the non-dominated
    trade-offs between the contested concepts themselves.

Usage:
    ecc = ECCParetoDQN(
        state_dim=110,
        num_actions=4,
        concept_configs=[
            {
                "name": "safety",
                "obj_indices": [0, 1],   # indices into full reward
                "ref_point": np.array([-1.0, -20.0]),
                "reward_min": np.array([0.0, -19.0]),
                "reward_max": np.array([124.0, -1.0]),
            },
            {
                "name": "fairness",
                "obj_indices": [2, 3],
                "ref_point": np.array([-1.0, -20.0]),
                "reward_min": np.array([0.0, -19.0]),
                "reward_max": np.array([124.0, -1.0]),
            },
        ],
        combination_mode="hypervolume",   # or "eum"
    )
"""

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume

from pareto_dqn import ParetoDQN


# ═══════════════════════════════════════════════════════════════════════════
#  EUM utility
# ═══════════════════════════════════════════════════════════════════════════

def expected_utility_metric(
    ref_point: np.ndarray,
    points: list,
    n_weights: int = 100,
) -> float:
    """Expected Utility Metric — E_{w~Unif(Δ)}[ max_{p∈PF} w·p ].

    Estimates by sampling weight vectors uniformly on the simplex.

    Args:
        ref_point: Reference point (unused by EUM itself but kept for API
                   symmetry with hypervolume).
        points:    List of array-like, each of shape (d,).
        n_weights: Number of weight samples.

    Returns:
        EUM scalar (higher is better).
    """
    if len(points) == 0:
        return 0.0
    pts = np.array(points)
    d = pts.shape[1]
    weights = np.random.dirichlet(np.ones(d), size=n_weights)  # (W, d)
    utilities = weights @ pts.T                                 # (W, M)
    return float(np.mean(np.max(utilities, axis=1)))


# ═══════════════════════════════════════════════════════════════════════════
#  ECC-Pareto-DQN
# ═══════════════════════════════════════════════════════════════════════════

class ECCParetoDQN:
    """Essentially Contested Concept Pareto-DQN.

    Each contested concept has its own full Pareto-DQN learner that
    operates on a *slice* of the reward vector.  Action selection
    combines the per-concept quality scores (HV or EUM) to balance
    exploration across all concepts.

    Args:
        state_dim:       Flat state dimension.
        num_actions:     Number of discrete actions.
        concept_configs: List of dicts, one per concept, each containing:
            - "name"        : str — human-readable label
            - "obj_indices" : List[int] — indices into the full reward vector
            - "ref_point"   : np.ndarray — HV reference for this concept
            - "reward_min"  : np.ndarray — per-obj min (for prefix sampling)
            - "reward_max"  : np.ndarray — per-obj max
        combination_mode: "hypervolume" or "eum".
        pdqn_kwargs:     Extra kwargs forwarded to each ParetoDQN learner
                         (gamma, lr_reward, lr_ndt, hidden, etc.).
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        concept_configs: List[dict],
        combination_mode: str = "hypervolume",
        **pdqn_kwargs,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_concepts = len(concept_configs)
        self.concept_configs = concept_configs
        self.combination_mode = combination_mode

        # Build one ParetoDQN per concept ---------------------------------
        self.learners: List[ParetoDQN] = []
        self.concept_names: List[str] = []
        self.obj_indices: List[List[int]] = []

        for cfg in concept_configs:
            self.concept_names.append(cfg["name"])
            self.obj_indices.append(cfg["obj_indices"])
            reward_dim = len(cfg["obj_indices"])

            learner = ParetoDQN(
                state_dim=state_dim,
                num_actions=num_actions,
                reward_dim=reward_dim,
                ref_point=np.array(cfg["ref_point"]),
                reward_min=np.array(cfg["reward_min"]),
                reward_max=np.array(cfg["reward_max"]),
                **pdqn_kwargs,
            )
            self.learners.append(learner)

        self.global_step = 0

    # ── Per-concept scoring ───────────────────────────────────────────────

    def _concept_score(self, concept_idx: int, state: np.ndarray,
                        action: int) -> float:
        """Score one action under one concept using the chosen metric."""
        learner = self.learners[concept_idx]
        qset = learner.predict_qset(state, action)

        if self.combination_mode == "hypervolume":
            return hypervolume(learner.ref_point, list(qset))
        elif self.combination_mode == "eum":
            return expected_utility_metric(learner.ref_point, list(qset))
        else:
            raise ValueError(f"Unknown combination_mode: {self.combination_mode}")

    # ── Combined scoring (mirrors ECC-PQL score_hypervolume) ──────────────

    def score_actions(self, state: np.ndarray) -> np.ndarray:
        """Compute combined scores for all actions.

        Score(a) = ∏_c  concept_score_c(s, a)

        The product encourages balanced performance across all concepts,
        exactly as in ECC-PQL.

        Returns:
            (num_actions,) numpy array.
        """
        per_concept = np.ones((self.num_concepts, self.num_actions))
        for c in range(self.num_concepts):
            for a in range(self.num_actions):
                per_concept[c, a] = self._concept_score(c, state, a)
        return np.prod(per_concept, axis=0)

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy over the combined score.

        Uses the first learner's epsilon (all share the same decay schedule).
        """
        if random.random() < self.learners[0].epsilon:
            return random.randrange(self.num_actions)

        scores = self.score_actions(state)
        return int(np.random.choice(np.flatnonzero(scores == scores.max())))

    # ── Training ──────────────────────────────────────────────────────────

    def observe(
        self,
        state: np.ndarray,
        action: int,
        full_reward: np.ndarray,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Store transition and update all concept learners.

        Each learner receives only the reward slice corresponding to its
        concept's objective indices.

        Args:
            full_reward: Full reward vector for ALL objectives.

        Returns:
            Dict of per-concept losses (or None if buffer too small).
        """
        all_losses: Dict[str, float] = {}
        for c, learner in enumerate(self.learners):
            concept_reward = full_reward[self.obj_indices[c]]
            losses = learner.observe(state, action, concept_reward,
                                     next_state, done)
            if losses is not None:
                name = self.concept_names[c]
                all_losses[f"{name}/loss_reward"] = losses["loss_reward"]
                all_losses[f"{name}/loss_ndt"] = losses["loss_ndt"]

        self.global_step += 1
        return all_losses if all_losses else None

    # ── Value-level Pareto front ──────────────────────────────────────────

    def get_value_pareto_front(self, state: np.ndarray) -> np.ndarray:
        """Return the Pareto front in (concept_0_score, …, concept_K_score) space.

        Each action maps to one point in this K-dimensional meta-space.
        The non-dominated subset is returned.  This is analogous to
        ECC-PQL's `get_value_pareto_front`.

        Returns:
            (M, num_concepts) numpy array of non-dominated value vectors.
        """
        value_pairs = []
        for a in range(self.num_actions):
            pair = np.array([
                self._concept_score(c, state, a)
                for c in range(self.num_concepts)
            ])
            value_pairs.append(pair)

        cands = {tuple(p) for p in value_pairs}
        nd = get_non_dominated(cands)
        return np.array(sorted(nd, key=lambda x: x[0]))

    def get_pareto_front_per_concept(
        self, state: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Return each concept learner's estimated Pareto front."""
        return {
            name: learner.get_pareto_front(state)
            for name, learner in zip(self.concept_names, self.learners)
        }

    # ── Policy tracking (analogous to ECC-PQL track_policy_*) ────────────

    def track_policy(
        self,
        concept_idx: int,
        target_vec: np.ndarray,
        env,
        tol: float = 1e-2,
        max_steps: int = 200,
    ) -> np.ndarray:
        """Track a specific policy point from a concept's Pareto front.

        At each step, pick the action whose Qset contains the point
        closest to *target_vec*, then update the target with the best
        matching NDt point (exactly as in PQL/ECC-PQL policy tracking).

        Args:
            concept_idx: Which concept to track.
            target_vec:  Target Qset point to follow (reward_dim of that concept).
            env:         Gymnasium-like environment.
            tol:         Early-stop tolerance.
            max_steps:   Safety limit.

        Returns:
            Accumulated (discounted) reward vector for the concept.
        """
        learner = self.learners[concept_idx]
        target = np.array(target_vec, dtype=np.float32)

        state, _ = env.reset()
        total_rew = np.zeros(learner.reward_dim)
        current_gamma = 1.0

        for _ in range(max_steps):
            best_dist = np.inf
            best_action = 0
            new_target = target

            for a in range(self.num_actions):
                qset = learner.predict_qset(state, a)
                for pt in qset:
                    dist = np.sum(np.abs(pt - target))
                    if dist < best_dist:
                        best_dist = dist
                        best_action = a
                        # Decompose: target ≈ R + γ·ndt_pt  →  ndt_pt for next step
                        s_t = torch.tensor(state, dtype=torch.float32,
                                           device=learner.device).unsqueeze(0)
                        a_t = torch.tensor([a], dtype=torch.long,
                                           device=learner.device)
                        with torch.no_grad():
                            r_hat = learner.reward_net(s_t, a_t).squeeze(0).cpu().numpy()
                        if learner.gamma > 0:
                            new_target = (np.array(pt) - r_hat) / learner.gamma
                        else:
                            new_target = np.zeros_like(target)

            state, reward, terminated, truncated, _ = env.step(best_action)
            concept_reward = reward[self.obj_indices[concept_idx]]
            total_rew += current_gamma * concept_reward
            current_gamma *= learner.gamma
            target = new_target

            if terminated or truncated:
                break

        return total_rew
