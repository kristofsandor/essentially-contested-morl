import gymnasium as gym
import numpy as np

# The ECC env emits a flattened [num_interps, net_reward_dim] reward matrix,
# laid out row-major (one row per ethical interpretation, each row a
# self-contained objective vector). For the reach-goal ECC env this is:
#   row 0 (deontological): [task, deont_help]
#   row 1 (utilitarian):   [task, util_help]
# -> [task, deont_help, task, util_help]
# These wrappers project that matrix down to a single net_reward_dim objective
# vector so scalar-interpretation (single-policy) agents can train on it. The
# interpretation count is read from the env (``num_interps``), defaulting to 2 to
# match the reach-goal ECC env, which does not set the attribute.
_DEFAULT_NUM_INTERPS = 2


def _interp_shape(env):
    """(num_interps, net_reward_dim) for the flattened reward matrix of ``env``."""
    num_interps = int(getattr(env.unwrapped, "num_interps", _DEFAULT_NUM_INTERPS))
    flat = env.unwrapped.reward_space.shape[0]
    assert flat % num_interps == 0, (
        f"reward_dim={flat} is not divisible by num_interps={num_interps}."
    )
    return num_interps, flat // num_interps


def _as_matrix(r, num_interps, net_reward_dim):
    """Reshape a flattened per-interpretation reward back into its rows."""
    return np.asarray(r, dtype=np.float32).reshape(num_interps, net_reward_dim)


class InterpProjectionWrapper(gym.RewardWrapper):
    """Expose only interpretation ``index``'s row of the reward matrix."""

    def __init__(self, env, index):
        super().__init__(env)
        self.index = index
        self.num_interps, self.net_reward_dim = _interp_shape(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=_as_matrix(rs.low, self.num_interps, self.net_reward_dim)[index],
            high=_as_matrix(rs.high, self.num_interps, self.net_reward_dim)[index],
        )

    def reward(self, reward):
        return _as_matrix(reward, self.num_interps, self.net_reward_dim)[self.index]


class DeontologicalWrapper(InterpProjectionWrapper):
    """Expose only the first interpretation (reach-goal: [task, deont_help])."""

    def __init__(self, env):
        super().__init__(env, 0)


class UtilitarianWrapper(InterpProjectionWrapper):
    """Expose only the second interpretation (reach-goal: [task, util_help])."""

    def __init__(self, env):
        super().__init__(env, 1)


class MeanInterpWrapper(gym.RewardWrapper):
    """Average the interpretations into a single net_reward_dim vector."""

    def __init__(self, env):
        super().__init__(env)
        self.num_interps, self.net_reward_dim = _interp_shape(env)
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=_as_matrix(rs.low, self.num_interps, self.net_reward_dim).mean(axis=0),
            high=_as_matrix(rs.high, self.num_interps, self.net_reward_dim).mean(axis=0),
        )

    def reward(self, reward):
        m = _as_matrix(reward, self.num_interps, self.net_reward_dim)
        return m.mean(axis=0).astype(np.float32)


class InterpWeightWrapper(gym.RewardWrapper):
    """Project interpretations by a weighted average at a fixed interp weight.

    reward -> sum_i interp_w[i] * row_i (a net_reward_dim vector). Used to evaluate
    a weight-conditioned ECC agent at a reference interpretation weight, so the env
    return lives in the agent's net_reward_dim objective space.
    """

    def __init__(self, env, interp_w):
        super().__init__(env)
        self.interp_w = np.asarray(interp_w, dtype=np.float32)
        self.num_interps, self.net_reward_dim = _interp_shape(env)
        assert len(self.interp_w) == self.num_interps, (
            f"interp_weight has length {len(self.interp_w)} but "
            f"num_interps={self.num_interps}."
        )
        rs = env.unwrapped.reward_space
        self.reward_space = gym.spaces.Box(
            low=(self.interp_w @ _as_matrix(rs.low, self.num_interps, self.net_reward_dim)).astype(np.float32),
            high=(self.interp_w @ _as_matrix(rs.high, self.num_interps, self.net_reward_dim)).astype(np.float32),
        )

    def reward(self, reward):
        m = _as_matrix(reward, self.num_interps, self.net_reward_dim)
        return (self.interp_w @ m).astype(np.float32)
