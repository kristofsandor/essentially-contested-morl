import gymnasium as gym
import numpy as np
from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport

from agent.ecc_envelope import ECCEnvelope
from agent.envelope import Envelope
from agent.pql import PQL
from agent.gpi_pd import GPIPD
from agent.ecc_gpi_pd import ECCGPIPD

from pathlib import Path

from agent.ucb_envelope import UCBEnvelope
from wrappers.matrix_to_vector import DeontologicalWrapper, UtilitarianWrapper


def _patch_linear_support_exact_arithmetic() -> None:
    """Route ``LinearSupport.compute_corner_weights`` through pycddlib's exact
    (gmp/fraction) backend.

    The env has pycddlib 3.0.2 while morl_baselines pins 2.1.6; the installed
    ``linear_support.py`` uses the 3.x API but its floating-point vertex
    enumeration segfaults on some CCS inputs (observed as a hang right after
    "Computing corner weights for CCS"). The exact backend is numerically robust
    and returns identical corner weights. Mirrors the original method's A/b
    construction and vertex filtering, swapping only the cdd number type.
    """
    from fractions import Fraction

    try:
        import cdd
        from cdd import gmp
    except ImportError:  # exact backend unavailable -> leave original in place
        return

    def compute_corner_weights(self):
        A = np.vstack(self.ccs)
        A = np.round(A, decimals=4)  # round to avoid numerical issues
        A = np.concatenate((A, -np.ones((A.shape[0], 1))), axis=1)
        A_plus = np.ones((1, A.shape[1]))
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)
        A_plus = -np.ones((1, A.shape[1]))
        A_plus[0, -1] = 0
        A = np.concatenate((A, A_plus), axis=0)
        for i in range(self.num_objectives):
            A_plus = np.zeros((1, A.shape[1]))
            A_plus[0, i] = -1
            A = np.concatenate((A, A_plus), axis=0)
        b = np.zeros(len(self.ccs) + 2 + self.num_objectives)
        b[len(self.ccs)] = 1
        b[len(self.ccs) + 1] = -1
        b = b.reshape((-1, 1))

        arr = np.hstack([b, -A])
        arr_frac = [[Fraction(float(x)).limit_denominator(10**6) for x in row] for row in arr]
        mat = gmp.matrix_from_array(arr_frac, rep_type=cdd.RepType.INEQUALITY)
        poly = gmp.polyhedron_from_matrix(mat)
        gens = gmp.copy_generators(poly)
        lin_set = set(gens.lin_set)

        corners = []
        for i, row in enumerate(gens.array):
            row = [float(x) for x in row]
            if row[0] != 1 or i in lin_set:  # skip rays / linear-set rows
                continue
            corner_weight = np.abs(np.array(row[1:-1]))  # drop homog. col + scalar var
            corner_weight /= corner_weight.sum()
            corners.append(corner_weight)
        return corners

    LinearSupport.compute_corner_weights = compute_corner_weights


_patch_linear_support_exact_arithmetic()


AGENTS = {
    "envelope": Envelope,
    "pql": PQL,
    "ucb_envelope": UCBEnvelope,
    "ecc_envelope": ECCEnvelope,
    "gpi_pd": GPIPD,
    "ecc_gpi_pd": ECCGPIPD,
}


def find_model_path(run_id):
    base = "results/reach_goal"
    # any folder under base dir that contains run_id (recursive, multiple level of folders and i search the lowest level one)
    for path in Path(base).rglob(f"*{run_id}*"):
        if path.is_dir():
            return path / "model.tar"


def make_agent(env, agent_config):
    if "algorithm" not in agent_config:
        agent_name = "envelope"
    else:
        agent_name = agent_config.pop("algorithm")
    if agent_name not in AGENTS:
        raise ValueError(f"unknown agent name {agent_name}")
    return AGENTS[agent_name](env=env, **agent_config)


def make_env(env_config) -> gym.Env:
    use_util = env_config.pop("utilitarian_wrapper", False)
    use_deont = env_config.pop("deontological_wrapper", False)
    env = gym.make(**env_config)
    if use_util:
        env = UtilitarianWrapper(env)
    elif use_deont:
        env = DeontologicalWrapper(env)
    reward_wrapped = env  # reward space the agent actually sees
    env = MORecordEpisodeStatistics(env)
    # MORecordEpisodeStatistics infers its accumulator dim from env.unwrapped, which
    # bypasses reward wrappers. Sync it to the (possibly projected) reward space.
    if use_util or use_deont:
        rdim = reward_wrapped.reward_space.shape[0]
        env.reward_dim = rdim
        env.rewards_shape = (rdim,)
    return env
