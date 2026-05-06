"""Lexicographic reward wrapper for essentially contested concept MORL.

Applies strict lexicographic ordering *within* each contested concept (e.g., Safety, Fairness)
by scalarizing the interpretations of each concept into a single value using an exponentially
separated weight vector.  The outer MORL algorithm (e.g., MORLD) then maintains a Pareto front
*between* the per-concept scalars, using a multi-policy approach.

Strict lexicographic ordering guarantee
----------------------------------------
Given a concept with k interpretations [o_1, o_2, ..., o_k] (in priority order) whose
per-step rewards lie in [r_min, r_max], setting

    M  >  (r_max - r_min) * T   ... for undiscounted finite-horizon T

(or, with discount factor gamma: M > (r_max - r_min) * (1 - gamma^T) / (1 - gamma))

ensures that the composite scalar

    lex_value = o_1 * M^(k-1) + o_2 * M^(k-2) + ... + o_k * M^0

preserves strict lexicographic preference: any policy that achieves a strictly higher
expected return on o_i will always score higher on lex_value, regardless of o_{i+1}, ..., o_k.

For binary rewards (0 or 1 per step) with episode length T, using M = T + 1 satisfies
the undiscounted bound and is the recommended default.
"""

from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class LexRewardWrapper(gym.RewardWrapper):
    """Wraps a multi-objective environment so that MORL operates over concepts rather
    than raw interpretations.

    Within each concept the interpretations are combined with strict lexicographic
    (big-M) weighting.  The resulting reward vector has one component per concept,
    which the outer multi-policy algorithm (e.g., MORLD/GPI-PD) uses to produce a
    Pareto front *between* concepts.

    Parameters
    ----------
    env:
        The wrapped multi-objective Gymnasium environment.  Must expose a
        ``reward_space`` attribute on its ``unwrapped`` env.
    concept_groups:
        Each element is a list of reward-vector indices that belong to one
        contested concept, listed in *descending priority order* (index 0 is the
        highest-priority interpretation).
        Example for MyFourRoom: ``[[0, 1], [2, 3]]`` groups
        [blue_triangle, blue_circle] as Concept 0 and [red_triangle, red_circle]
        as Concept 1.
    lex_scale:
        The big-M multiplier.  Must be strictly greater than the maximum possible
        total (discounted) return of any *single* interpretation over one episode.
        Defaults to ``max_episode_steps + 1`` when a ``TimeLimit`` spec is
        available, otherwise 10.
    reward_bounds:
        Optional ``(low, high)`` tuple applied to the new reward space.  Defaults
        to ``(-inf, inf)``.

    Examples
    --------
    >>> env = gym.wrappers.TimeLimit(gym.make("my-four-room-v0"), max_episode_steps=8)
    >>> wrapped = LexRewardWrapper(env, concept_groups=[[0, 1], [2, 3]])
    >>> wrapped.reward_space.shape
    (2,)
    """

    def __init__(
        self,
        env: gym.Env,
        concept_groups: List[List[int]],
        lex_scale: Optional[float] = None,
        reward_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(env)

        self.concept_groups = concept_groups
        self.num_concepts = len(concept_groups)

        # Determine lex_scale automatically when not provided
        if lex_scale is None:
            if (
                hasattr(env, "spec")
                and env.spec is not None
                and hasattr(env.spec, "max_episode_steps")
                and env.spec.max_episode_steps is not None
            ):
                lex_scale = float(env.spec.max_episode_steps) + 1.0
            else:
                lex_scale = 10.0
        self.lex_scale = float(lex_scale)

        # Pre-compute the weight vector for each concept so we avoid recomputing
        # at every step.  For a concept of size k the weights are:
        #   [M^(k-1), M^(k-2), ..., M^1, M^0]
        self._concept_weights: List[np.ndarray] = []
        for group in concept_groups:
            k = len(group)
            weights = np.array(
                [self.lex_scale ** (k - 1 - i) for i in range(k)], dtype=np.float64
            )
            self._concept_weights.append(weights)

        # Update the reward_space on the unwrapped environment so that downstream
        # MORL algorithms (which read reward_space.shape) see the correct dimensionality.
        low, high = (-np.inf, np.inf) if reward_bounds is None else reward_bounds
        new_reward_space = Box(
            low=np.full(self.num_concepts, low, dtype=np.float32),
            high=np.full(self.num_concepts, high, dtype=np.float32),
            shape=(self.num_concepts,),
            dtype=np.float32,
        )
        self.env.unwrapped.reward_space = new_reward_space
        self.env.unwrapped.reward_dim = self.num_concepts

    # ------------------------------------------------------------------
    # gym.RewardWrapper interface
    # ------------------------------------------------------------------

    def reward(self, reward: np.ndarray) -> np.ndarray:
        """Transform the raw multi-objective reward into a per-concept scalar.

        Parameters
        ----------
        reward:
            Raw reward vector from the wrapped environment.

        Returns
        -------
        np.ndarray
            Shape ``(num_concepts,)`` — one strict-lex scalar per concept.
        """
        reward = np.asarray(reward, dtype=np.float64)
        out = np.empty(self.num_concepts, dtype=np.float32)
        for i, (group, weights) in enumerate(
            zip(self.concept_groups, self._concept_weights)
        ):
            out[i] = float(np.dot(weights, reward[group]))
        return out
