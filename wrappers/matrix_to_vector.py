import gymnasium as gym
import numpy as np

# The ECC env emits a flattened [num_interps, net_reward_dim] reward matrix,
# laid out row-major (one row per ethical interpretation, each row a
# self-contained objective vector [task, help]):
#   row 0 (deontological): [task, deont_help]
#   row 1 (utilitarian):   [task, util_help]
# -> [task, deont_help, task, util_help]
# These wrappers project that matrix down to a single 2-objective vector so
# scalar-interpretation (single-policy) agents can train on it.
_NUM_INTERPS = 2
_NET_REWARD_DIM = 2


def _as_matrix(r):
    """Reshape a flattened per-interpretation reward back into its rows."""
    return np.asarray(r, dtype=np.float32).reshape(_NUM_INTERPS, _NET_REWARD_DIM)


class DeontologicalWrapper(gym.RewardWrapper):
    """Expose only the deontological interpretation: [task, deont_help]."""

    def __init__(self, env):
        super().__init__(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=_as_matrix(rs.low)[0], high=_as_matrix(rs.high)[0]
        )

    def reward(self, reward):
        return _as_matrix(reward)[0]


class UtilitarianWrapper(gym.RewardWrapper):
    """Expose only the utilitarian interpretation: [task, util_help]."""

    def __init__(self, env):
        super().__init__(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=_as_matrix(rs.low)[1], high=_as_matrix(rs.high)[1]
        )

    def reward(self, reward):
        return _as_matrix(reward)[1]


class MeanInterpWrapper(gym.RewardWrapper):
    """Average the interpretations: [task, mean(deont_help, util_help)]."""

    def __init__(self, env):
        super().__init__(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=_as_matrix(rs.low).mean(axis=0),
            high=_as_matrix(rs.high).mean(axis=0),
        )

    def reward(self, reward):
        return _as_matrix(reward).mean(axis=0).astype(np.float32)
