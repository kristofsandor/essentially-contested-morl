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
from use_cases.firefighters_use_case.constants import (
    ACTION_AGGRESSIVE_FIRE_SUPPRESSION,
    ACTION_ASSESS_AND_PLAN,
    ACTION_CONTAIN_FIRE,
    ACTION_COORDINATE_WITH_OTHER_AGENCIES,
    ACTION_EVACUATE_OCCUPANTS,
)

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


def _by_action(mapping: dict, n_actions: int = 5) -> np.ndarray:
    """Build a length-n_actions reward vector from an {action_id: reward} map."""
    arr = np.zeros(n_actions, dtype=np.float32)
    for action_id, value in mapping.items():
        arr[action_id] = value
    return arr


def _per_action(rewards: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Grounding that assigns a fixed reward per action, then reapplies penalties.

    Each action gets a flat reward from ``rewards`` in every state. The hard
    override penalty of the underlying value (cells the base grounding forces to
    -1.0: nonsensical actions and firefighter incapacitation) is then stamped
    back on, so all interpretations share the exact same guardrails and only
    contest the positive region.
    """
    rewards = np.asarray(rewards, dtype=np.float32)

    def fn(base: np.ndarray) -> np.ndarray:
        out = np.broadcast_to(rewards, base.shape).astype(np.float32).copy()
        out[np.isclose(base, -1.0)] = -1.0
        return out

    return fn


# The groundings keep the base professionalism/proximity character so the two
# values still trade off *within* an interpretation: professionalism rewards the
# methodical, by-the-book actions (assess, coordinate, contain) and proximity
# rewards the direct, hands-on actions (evacuate, aggressive suppression). The
# interpretation (rescue vs fire) then shifts *which* methodical and direct
# actions are emphasised, which makes the interpretations disagree with each
# other. This dual structure gives each interpretation a non-trivial Pareto front
# and produces large cross-interpretation regret under the contested dynamics.

# Two rival readings of professionalism (the methodical value).
# Rescue: plan and prepare the rescue (assess, coordinate). Fire: control the
# blaze by the book (contain).
_PF_RESCUE = _per_action(
    _by_action(
        {
            ACTION_EVACUATE_OCCUPANTS: 0.3,
            ACTION_CONTAIN_FIRE: 0.4,
            ACTION_AGGRESSIVE_FIRE_SUPPRESSION: 0.0,
            ACTION_COORDINATE_WITH_OTHER_AGENCIES: 0.7,
            ACTION_ASSESS_AND_PLAN: 1.0,
        }
    )
)
_PF_FIRE = _per_action(
    _by_action(
        {
            ACTION_EVACUATE_OCCUPANTS: 0.0,
            ACTION_CONTAIN_FIRE: 1.0,
            ACTION_AGGRESSIVE_FIRE_SUPPRESSION: 0.3,
            ACTION_COORDINATE_WITH_OTHER_AGENCIES: 0.5,
            ACTION_ASSESS_AND_PLAN: 0.7,
        }
    )
)

# Two rival readings of proximity (the direct value).
# People: rush to the occupants (evacuate). Fire: rush to the blaze (aggressive).
# Both keep the deliberative actions negative, so proximity stays distinct from
# professionalism rather than collinear with it.
_PX_PEOPLE = _per_action(
    _by_action(
        {
            ACTION_EVACUATE_OCCUPANTS: 1.0,
            ACTION_CONTAIN_FIRE: 0.1,
            ACTION_AGGRESSIVE_FIRE_SUPPRESSION: 0.2,
            ACTION_COORDINATE_WITH_OTHER_AGENCIES: -0.3,
            ACTION_ASSESS_AND_PLAN: -0.3,
        }
    )
)
_PX_FIRE = _per_action(
    _by_action(
        {
            ACTION_EVACUATE_OCCUPANTS: 0.1,
            ACTION_CONTAIN_FIRE: 0.3,
            ACTION_AGGRESSIVE_FIRE_SUPPRESSION: 1.0,
            ACTION_COORDINATE_WITH_OTHER_AGENCIES: -0.3,
            ACTION_ASSESS_AND_PLAN: -0.3,
        }
    )
)


_NAMED_MODES = {
    "graded": _graded,
    "default": _graded,
    "idealist": _idealist,
    "strict": _idealist,
    # action-based rival readings (see tables / docstring)
    "pf_rescue": _PF_RESCUE,
    "pf_fire": _PF_FIRE,
    "px_people": _PX_PEOPLE,
    "px_fire": _PX_FIRE,
}


# Ready-made interpretation sets. Each maps to (interpretations, labels). Pass
# the name to ``ECCFireFightersEnvMO.from_preset``.
INTERPRETATION_PRESETS = {
    # original graded vs strict/idealist reading of both values
    "graded_vs_idealist": (
        [("graded", "graded"), ("idealist", "idealist")],
        ["graded", "idealist"],
    ),
    # contest professionalism only; proximity held at the graded ground truth
    "professionalism_contest": (
        [("pf_rescue", "graded"), ("pf_fire", "graded")],
        ["rescue_pro", "fire_pro"],
    ),
    # contest proximity only; professionalism held at the graded ground truth
    "proximity_contest": (
        [("graded", "px_people"), ("graded", "px_fire")],
        ["people_prox", "fire_prox"],
    ),
    # two internally coherent stances: rescue-minded vs fire-minded
    "rescue_vs_fire": (
        [("pf_rescue", "px_people"), ("pf_fire", "px_fire")],
        ["rescue_minded", "fire_minded"],
    ),
    # maximally tense pairing: manner and target pull opposite ways
    "crossed": (
        [("pf_rescue", "px_fire"), ("pf_fire", "px_people")],
        ["rescue_pro_fire_prox", "fire_pro_people_prox"],
    ),
    # all five readings together, for overlaying every Pareto front in one figure.
    # Intended for analysis / front plotting rather than training (num_interps = 5).
    "all_five": (
        [
            ("graded", "graded"),
            ("pf_rescue", "px_people"),
            ("pf_fire", "px_fire"),
            ("pf_rescue", "px_fire"),
            ("pf_fire", "px_people"),
        ],
        [
            "graded",
            "rescue_minded",
            "fire_minded",
            "crossed_rescuePF_firePX",
            "crossed_firePF_peoplePX",
        ],
    ),
    "non_crossed": (
        [
            ("graded", "graded"),
            ("pf_rescue", "px_people"),
            ("pf_fire", "px_fire"),
        ],
        [
            "graded",
            "rescue_minded",
            "fire_minded",
        ],
    ),
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
            interpretation_labels = [f"interp_{i}" for i in range(self.num_interps)]
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

    @classmethod
    def from_preset(cls, preset: str, **kwargs) -> "ECCFireFightersEnvMO":
        """Build the env from a named entry in ``INTERPRETATION_PRESETS``.

        Example:
            env = ECCFireFightersEnvMO.from_preset("rescue_vs_fire", horizon=50)
        """
        try:
            interpretations, labels = INTERPRETATION_PRESETS[preset]
        except KeyError as exc:
            raise ValueError(
                f"Unknown preset {preset!r}. "
                f"Available: {sorted(INTERPRETATION_PRESETS)}."
            ) from exc
        return cls(
            interpretations=interpretations,
            interpretation_labels=labels,
            **kwargs,
        )


class ContestedFireFightersEnvMO(ECCFireFightersEnvMO):
    """ECC firefighters with time pressure, so the interpretations actually conflict.

    The plain ECC env has a flaw for studying interpretation robustness: with a
    50-step horizon the optimal trajectory can rescue everyone *and* extinguish
    the fire, so re-weighting actions across interpretations costs nothing and a
    policy trained on one interpretation is near-optimal for all of them.

    This variant adds two passive dynamics plus a tight horizon, creating a real
    opportunity cost between rescuing and firefighting:

      * fire spreads (+1 intensity, capped) on any step that is not a suppression
        action (Contain / Aggressive), while a fire is active;
      * one occupant is lost (occupancy -1) on any step that is not Evacuate.

    With a short horizon you cannot do both, so rescue-leaning and fire-leaning
    interpretations induce genuinely different optimal policies, and committing to
    the wrong interpretation incurs measurable regret.

    The reward function is unchanged: ``calculate_rewards`` depends only on the
    current state, the action, and ``next_state[MEDICAL]``, and the drift never
    alters MEDICAL, so only the transition dynamics and the horizon change.

    Args:
        horizon: episode length cap (default 5, tight enough for the trade-off).
        fire_spreads / occupants_flee: toggle each pressure independently.
    """

    def __init__(
        self,
        *args,
        horizon: int = 5,
        fire_spreads: bool = True,
        occupants_flee: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fire_spreads = fire_spreads
        self.occupants_flee = occupants_flee

        re = self.real_env
        base_transition = re.transition  # original deterministic dynamics
        fire_dim = int(re.state_space.nvec[1])  # number of fire levels

        def contested(state, action):
            ns = np.asarray(base_transition(state, action)).copy()
            fi, oc = ns[1], ns[2]
            if (
                self.fire_spreads
                and action
                not in (ACTION_CONTAIN_FIRE, ACTION_AGGRESSIVE_FIRE_SUPPRESSION)
                and 0 < fi < fire_dim - 1
            ):
                ns[1] = fi + 1
            if self.occupants_flee and action != ACTION_EVACUATE_OCCUPANTS and oc > 0:
                ns[2] = oc - 1
            return ns

        # route step() through the contested dynamics too
        re.transition = contested

        # rebuild only the transition matrix; rewards are identical (see docstring)
        T = np.zeros_like(self.transition_matrix)
        for s in range(self.n_states):
            st = re.translate(s)
            if not re.is_done(st):
                for a in range(self.action_space.n):
                    T[s, a, re.encrypt(contested(st, a))] = 1.0
            else:
                for a in range(self.action_space.n):
                    T[s, a, s] = 1.0
        self.transition_matrix = T
        self.horizon = horizon
