"""Essentially-contested-concept (ECC) variant of the firefighters MO env.

The base ``FireFightersEnvMO`` exposes two values, professionalism and
proximity, as a single ``(professionalism, proximity)`` reward vector. This ECC
wrapper keeps those two values but lets each be *grounded* in several different
ways, treating each grounding as a separate interpretation of the same value.

The reward is emitted as a flattened ``[num_interps, net_reward_dim]`` matrix,
row-major, exactly the contract the ECC agents expect:

    net_reward_dim = 2            # (professionalism, proximity)
    num_interps    = len(interpretations)
    reward_dim     = num_interps * 2

For each step the env returns

    [interp0_pf, interp0_px, interp1_pf, interp1_px, ...]

``ECCEnvelope`` reshapes this back into rows and trains one Q-net per
interpretation; ``ECCGPIPD`` collapses the rows with a conditioning interp
weight. Both read ``num_interps`` from their own constructor, and the
``matrix_to_vector`` wrappers read ``num_interps`` off this env, so it is set as
an attribute below.
"""

from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from gymnasium import spaces

from env.firefighters_env_mo import FeatureSelectionFFEnv, FireFightersEnvMO

# A per-value interpretation spec is either a named mode or a callable that maps
# a base (S, A) reward matrix to a regrounded (S, A) matrix of the same shape.
InterpSpec = Union[str, Callable[[np.ndarray], np.ndarray]]


def _graded(base: np.ndarray) -> np.ndarray:
    """Literal ground-truth grounding: the value as the env computes it."""
    return base.copy()


def _idealist(base: np.ndarray) -> np.ndarray:
    """Strict / all-or-nothing grounding.

    Any action that contributes positively to the value is treated as fully
    aligned (reward 1.0); non-positive entries are left untouched. Mirrors the
    'professionalist' / 'proximitier' variants in ``obtain_grounding``.
    """
    out = base.copy()
    out[out > 0] = 1.0
    return out


_NAMED_MODES = {
    "graded": _graded,
    "default": _graded,
    "idealist": _idealist,
    "strict": _idealist,
}


def _resolve(spec: InterpSpec) -> Callable[[np.ndarray], np.ndarray]:
    """Turn an interpretation spec into a (S, A) -> (S, A) grounding function."""
    if callable(spec):
        return spec
    try:
        return _NAMED_MODES[spec]
    except KeyError as exc:
        raise ValueError(
            f"Unknown interpretation mode {spec!r}. "
            f"Use one of {sorted(_NAMED_MODES)} or pass a callable."
        ) from exc


class ECCFireFightersEnvMO(FireFightersEnvMO):
    """Firefighters MO env with multiple groundings of the two values.

    Args:
        interpretations: list of ``(pf_spec, px_spec)`` pairs, one per
            interpretation. Each spec is a named mode ('graded' / 'idealist') or
            a callable mapping a base (S, A) reward matrix to a regrounded one.
            Defaults to two interpretations: the graded ground truth and the
            idealist (strict) reading of both values.
        interpretation_labels: optional human-readable names, for logging.
        **kwargs: forwarded to ``FireFightersEnvMO`` (feature_selection,
            horizon, initial_state_distribution, ...).
    """

    def __init__(
        self,
        interpretations: Sequence[Tuple[InterpSpec, InterpSpec]] = None,
        interpretation_labels: Sequence[str] = None,
        feature_selection=FeatureSelectionFFEnv.ONE_HOT_FEATURES,
        **kwargs,
    ):
        super().__init__(feature_selection=feature_selection, **kwargs)

        if interpretations is None:
            interpretations = [
                ("graded", "graded"),
                ("idealist", "idealist"),
            ]
        self.interpretations = list(interpretations)
        self.num_interps = len(self.interpretations)
        assert self.num_interps >= 1, "Need at least one interpretation."

        if interpretation_labels is None:
            interpretation_labels = [
                f"interp_{i}" for i in range(self.num_interps)
            ]
        assert len(interpretation_labels) == self.num_interps
        self.interpretation_labels = list(interpretation_labels)

        # Base (S, A) reward matrices for each value, as built by the parent.
        base_pf = self.reward_matrix_per_va_dict[(1.0, 0.0)]  # professionalism
        base_px = self.reward_matrix_per_va_dict[(0.0, 1.0)]  # proximity

        # Per interpretation, reground each value column -> (S, A, 2).
        groundings: List[np.ndarray] = []
        for pf_spec, px_spec in self.interpretations:
            g = np.zeros(
                (self.n_states, self.action_space.n, self.n_values),
                dtype=np.float32,
            )
            g[:, :, 0] = _resolve(pf_spec)(base_pf)
            g[:, :, 1] = _resolve(px_spec)(base_px)
            groundings.append(g)

        # (S, A, num_interps, 2) -> flatten last two axes row-major to
        # (S, A, num_interps * 2). For a fixed (s, a) this is
        # [i0_pf, i0_px, i1_pf, i1_px, ...], matching the agents' reshape.
        tensor = np.stack(groundings, axis=2)
        self.reward_matrix = tensor.reshape(
            self.n_states, self.action_space.n, self.num_interps * self.n_values
        ).astype(np.float32)

        # Keep the original two-value matrix available for reference / debugging.
        self.reward_matrix_per_interp = tensor  # (S, A, num_interps, 2)

        # reward_dim is the flat length; net_reward_dim stays 2.
        self.reward_dim = self.num_interps * self.n_values
        self.net_reward_dim = self.n_values

        flat = self.reward_matrix.reshape(-1, self.reward_dim)
        self.reward_space = spaces.Box(
            low=flat.min(axis=0).astype(np.float32),
            high=flat.max(axis=0).astype(np.float32),
            dtype=np.float32,
        )
