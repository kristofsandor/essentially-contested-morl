"""Observation wrappers for the FireRescue env.

The base FireRescueEnv exposes a nested gymnasium ``Dict`` observation, which
is convenient for rendering and inspection but not compatible with most
morl_baselines algorithms (Envelope, GPI-PD, CAPQL, MPMOQL, ...) that assume
a flat ``Box`` observation.

``FlattenFireRescueObs`` converts the dict into a single 1-D ``float32`` Box
in a fixed order so downstream code can rely on the layout.

Layout of the flattened vector (length L)::

    [agent_row, agent_col,                                     # 2
     fire_map_flattened (size*size),                           # S
     entity_positions_flattened (num_entities * 2),            # 2E
     entity_types (num_entities),                              # E
     entity_ages (num_entities),                               # E
     entity_vulnerability (num_entities),                      # E
     entity_rescued (num_entities),                            # E
     entity_damage (num_entities),                             # E
     diamond_positions_flattened (num_diamonds * 2),           # 2D
     diamond_collected (num_diamonds)]                         # D

so L = 2 + S + 6E + 3D, where S=size*size, E=num_entities, D=num_diamonds.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenFireRescueObs(gym.ObservationWrapper):
    """Flatten the FireRescue Dict observation into a single Box vector.

    The wrapper assumes the wrapped env's observation_space matches the shape
    produced by :class:`env.fire_rescue.FireRescueEnv`. It does not modify
    the action space or reward space.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # We need to know the sub-shapes to compute the flattened length and
        # to safely re-order in `observation`. Pull them from the underlying
        # env rather than from the (possibly already-wrapped) observation_space
        # so we are robust to other wrappers.
        unwrapped = env.unwrapped
        self._size = unwrapped.size
        self._num_entities = unwrapped.num_entities
        self._num_diamonds = unwrapped.num_diamonds

        flat_len = (
            2  # agent
            + self._size * self._size  # fire map
            + 2 * self._num_entities  # entity positions
            + 5 * self._num_entities  # types, ages, vulnerability, rescued, damage
            + 2 * self._num_diamonds  # diamond positions
            + self._num_diamonds  # diamond collected
        )

        # All components are non-negative integers or floats in [0, 1]/[0, 100],
        # so a generous upper bound suffices. We use float32 throughout.
        high = float(max(self._size, 100))  # ages can reach 100
        self.observation_space = spaces.Box(
            low=0.0,
            high=high,
            shape=(flat_len,),
            dtype=np.float32,
        )

    def observation(self, obs):
        agent = np.asarray(obs["agent"], dtype=np.float32).reshape(-1)
        fire = np.asarray(obs["fire"], dtype=np.float32).reshape(-1)
        ents = obs["entities"]
        diamonds = obs["diamonds"]

        ent_pos = np.asarray(ents["positions"], dtype=np.float32).reshape(-1)
        ent_types = np.asarray(ents["types"], dtype=np.float32).reshape(-1)
        ent_ages = np.asarray(ents["ages"], dtype=np.float32).reshape(-1)
        ent_vul = np.asarray(ents["vulnerability"], dtype=np.float32).reshape(-1)
        ent_resc = np.asarray(ents["rescued"], dtype=np.float32).reshape(-1)
        ent_dmg = np.asarray(ents["damage"], dtype=np.float32).reshape(-1)

        diam_pos = np.asarray(diamonds["positions"], dtype=np.float32).reshape(-1)
        diam_col = np.asarray(diamonds["collected"], dtype=np.float32).reshape(-1)

        return np.concatenate(
            [
                agent,
                fire,
                ent_pos,
                ent_types,
                ent_ages,
                ent_vul,
                ent_resc,
                ent_dmg,
                diam_pos,
                diam_col,
            ]
        ).astype(np.float32)
