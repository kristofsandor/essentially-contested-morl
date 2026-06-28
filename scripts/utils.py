import gymnasium as gym
import numpy as np
from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport

from agent.pql import PQL
from agent.ecc_envelope import ECCEnvelope
from agent.envelope import Envelope
from agent.gpi_pd import GPIPD
from agent.ecc_gpi_pd import ECCGPIPD

from pathlib import Path

from agent.ucb_envelope import UCBEnvelope
from wrappers.matrix_to_vector import (
    DeontologicalWrapper,
    InterpProjectionWrapper,
    InterpWeightWrapper,
    UtilitarianWrapper,
)


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


def _patch_reward_dim_honors_wrappers() -> None:
    """Size ``MOAgent.reward_dim`` from the outermost reward wrapper, not the raw env.

    morl_baselines' ``extract_env_info`` reads ``env.unwrapped.reward_space``, which
    bypasses our per-interpretation reward wrappers (Deontological/Utilitarian/
    InterpWeight). A single-policy agent trained on a projected env would then be
    sized to the raw multi-interpretation reward (e.g. the 4-D [task, deont_help,
    task, util_help] matrix) but fed 2-D projected rewards, crashing on the first
    replay-buffer add. Walk outward to the first wrapper that defines ``reward_space``
    (the space the agent actually sees), mirroring ``eval_envelope.wrapped_reward_space``.
    With no reward wrapper present the outermost space is the unwrapped one, so ECC /
    raw-env training is unchanged.
    """
    from morl_baselines.common.morl_algorithm import MOAgent

    _orig_extract = MOAgent.extract_env_info

    def extract_env_info(self, env):
        _orig_extract(self, env)
        if env is None:
            return
        e = env
        while e is not None:
            rs = e.__dict__.get("reward_space")  # __dict__ avoids triggering proxies
            if rs is not None:
                self.reward_dim = rs.shape[0]
                break
            e = getattr(e, "env", None)

    MOAgent.extract_env_info = extract_env_info


_patch_reward_dim_honors_wrappers()


def _patch_wandb_media_tmp_dir() -> None:
    """Recreate wandb's media staging dir on demand so long runs don't crash.

    wandb stages logged media (``wandb.Table``/``wandb.Image``) in a process-wide
    temp dir (``MEDIA_TMP``) created once at import under the system temp folder.
    On long (e.g. overnight) runs, Windows temp cleanup can delete that dir, so the
    next table/image write fails with ``FileNotFoundError`` inside ``bind_to_run`` —
    killing training over a non-essential log (observed in ``log_all_multi_policy_metrics``
    logging ``eval/front``). Wrap the media ``bind_to_run`` methods to ``makedirs`` the
    staging dir first. No-ops if wandb's internals differ from what we expect.
    """
    import importlib
    import os

    try:
        table_mod = importlib.import_module("wandb.sdk.data_types.table")
        media_mod = importlib.import_module("wandb.sdk.data_types.base_types.media")
        media_tmp = getattr(table_mod, "MEDIA_TMP")
        targets = [media_mod.Media, table_mod.Table]
    except Exception:
        return

    def _wrap(cls):
        # Only wrap classes that define their own bind_to_run (so we patch both the
        # base Media and Table's override), and guard against double-patching.
        if "bind_to_run" not in cls.__dict__ or getattr(cls.bind_to_run, "_media_tmp_patched", False):
            return
        orig = cls.__dict__["bind_to_run"]

        def bind_to_run(self, *args, **kwargs):
            try:
                os.makedirs(media_tmp.name, exist_ok=True)
            except Exception:
                pass
            return orig(self, *args, **kwargs)

        bind_to_run._media_tmp_patched = True
        cls.bind_to_run = bind_to_run

    for cls in targets:
        _wrap(cls)


_patch_wandb_media_tmp_dir()


AGENTS = {
    "envelope": Envelope,
    "pql": PQL,
    "ucb_envelope": UCBEnvelope,
    "ecc_envelope": ECCEnvelope,
    "gpi_pd": GPIPD,
    "ecc_gpi_pd": ECCGPIPD,
}


def interp_label_list(num_interps, env=None):
    """Human-readable labels for each ECC interpretation.

    Prefers the env's own ``interpretation_labels`` when present (e.g. the
    firefighters ECC env names them 'graded'/'idealist'), so eval outputs are
    labelled meaningfully. Otherwise: the reach-goal ECC env uses exactly two
    interpretations, so preserve its deontological/utilitarian output filenames for
    backward compatibility; for any other count fall back to ``interp_0 … interp_{n-1}``.
    """
    if env is not None:
        labels = getattr(getattr(env, "unwrapped", env), "interpretation_labels", None)
        if labels is not None and len(labels) == num_interps:
            return list(labels)
    if num_interps == 2:
        return ["deontological", "utilitarian"]
    return [f"interp_{i}" for i in range(num_interps)]


def find_model_path(run_id):
    base = "results/firefighters-mo-ecc-v0"
    # any folder under base dir that contains run_id (recursive, multiple level of folders and i search the lowest level one)
    for path in Path(base).rglob(f"*{run_id}*"):
        if path.is_dir():
            return path / "model.tar"
    raise ValueError(f"Could not find model path for run_id {run_id} under {base}")


def make_agent(env, agent_config):
    agent_name = agent_config.pop("algorithm")
    if agent_name not in AGENTS:
        raise ValueError(f"unknown agent name {agent_name}")
    return AGENTS[agent_name](env=env, **agent_config)


def make_env(env_config) -> gym.Env:
    # interp_index projects the flattened reward matrix onto a single interpretation
    # (the num_interps-generic path used by the per-interpretation eval). The boolean
    # flags are kept for backward compatibility: deontological == index 0, utilitarian
    # == index 1.
    interp_index = env_config.pop("interp_index", None)
    use_util = env_config.pop("utilitarian_wrapper", False)
    use_deont = env_config.pop("deontological_wrapper", False)
    use_interp_weight = env_config.pop("interp_weight_wrapper", False)
    # weight over interpretations; InterpWeightWrapper does interp_w @ matrix,
    # so a scalar would 0-d-matmul-fail. Default to equal weighting.
    interp_w = env_config.pop("interp_weight", [0.5, 0.5])
    env = gym.make(**env_config)
    if interp_index is not None:
        env = InterpProjectionWrapper(env, interp_index)
    elif use_util:
        env = UtilitarianWrapper(env)
    elif use_deont:
        env = DeontologicalWrapper(env)
    elif use_interp_weight:
        env = InterpWeightWrapper(env, interp_w)
    reward_wrapped = env  # reward space the agent actually sees
    env = MORecordEpisodeStatistics(env)
    # MORecordEpisodeStatistics infers its accumulator dim from env.unwrapped, which
    # bypasses reward wrappers. Sync it to the (possibly projected) reward space.
    if interp_index is not None or use_util or use_deont or use_interp_weight:
        rdim = reward_wrapped.reward_space.shape[0]
        env.reward_dim = rdim
        env.rewards_shape = (rdim,)
    return env
