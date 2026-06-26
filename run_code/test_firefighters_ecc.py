"""Smoke test for the ECC firefighters env and the ECC agents.

Checks, without any wandb / full training loop:
  1. the env emits a flattened [num_interps, net_reward_dim] reward,
  2. each step's reward reshapes cleanly into per-interpretation rows,
  3. ECCEnvelope can ingest it and run a few finite-loss updates,
  4. ECCGPIPD can ingest it and run a few updates, then act.

Run from the repo root:  python -m run_code.test_firefighters_ecc
"""

import numpy as np

from env.firefighters_ecc import ECCFireFightersEnvMO
from env.firefighters_env_mo import FeatureSelectionFFEnv


def _fill_buffer(env, agent, n_transitions, seed=0):
    """Step the env with random actions, pushing transitions into the buffer."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_transitions):
        action = int(rng.integers(env.action_space.n))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.replay_buffer.add(obs, action, reward, next_obs, terminated)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs


def test_env_reward_shape():
    env = ECCFireFightersEnvMO(
        interpretations=[("graded", "graded"), ("idealist", "idealist")],
        feature_selection=FeatureSelectionFFEnv.ONE_HOT_FEATURES,
        horizon=20,
    )
    num_interps = env.num_interps
    net_reward_dim = env.net_reward_dim
    assert num_interps == 2
    assert net_reward_dim == 2
    assert env.reward_space.shape == (num_interps * net_reward_dim,)
    assert env.reward_dim == num_interps * net_reward_dim

    obs, _ = env.reset(seed=1)
    for _ in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        reward = np.asarray(reward)
        assert reward.shape == (num_interps * net_reward_dim,)
        matrix = reward.reshape(num_interps, net_reward_dim)
        assert np.all(np.isfinite(matrix))
        if terminated or truncated:
            obs, _ = env.reset()

    # The graded and idealist groundings should actually differ somewhere,
    # otherwise the interpretations are degenerate.
    g = env.reward_matrix_per_interp  # (S, A, num_interps, 2)
    assert not np.allclose(g[:, :, 0, :], g[:, :, 1, :]), (
        "graded and idealist groundings are identical everywhere"
    )
    print("[ok] env reward shape, reshape, and distinct groundings")
    return env


def test_envelope_updates():
    from agent.ecc_envelope import ECCEnvelope

    env = ECCFireFightersEnvMO(horizon=20)
    agent = ECCEnvelope(
        env,
        num_interps=env.num_interps,
        interp_weight=np.array([0.5, 0.5], dtype=np.float32),
        net_arch=[32, 32],
        batch_size=8,
        learning_starts=0,
        num_sample_w=2,
        per=True,
        log=False,
        seed=0,
    )
    assert agent.net_reward_dim == env.net_reward_dim
    assert agent.flat_reward_dim == env.reward_dim

    _fill_buffer(env, agent, n_transitions=200, seed=0)
    agent.global_step = agent.learning_starts + 1
    for _ in range(5):
        agent.update()
    assert np.isfinite(agent._last_loss), f"non-finite loss {agent._last_loss}"

    obs, _ = env.reset(seed=2)
    w = np.array([0.7, 0.3], dtype=np.float32)
    iw = np.array([0.5, 0.5], dtype=np.float32)
    action = agent.eval(obs, w, iw)
    assert 0 <= int(action) < env.action_space.n
    print(f"[ok] ECCEnvelope: 5 updates, last loss={agent._last_loss:.4f}, "
          f"greedy action={int(action)}")


def test_gpipd_updates():
    import torch as th
    from agent.ecc_gpi_pd import ECCGPIPD

    env = ECCFireFightersEnvMO(horizon=20)
    agent = ECCGPIPD(
        env,
        num_interps=env.num_interps,
        eval_interp_weight=np.array([0.5, 0.5], dtype=np.float32),
        net_arch=[32, 32, 32, 32],
        batch_size=8,
        learning_starts=0,
        gradient_updates=1,
        dyna=False,       # keep it light; no dynamics ensemble rollouts
        gpi_pd=False,
        per=True,
        log=False,
        seed=0,
    )
    assert agent.net_reward_dim == env.net_reward_dim

    agent.set_weight_support([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])
    _fill_buffer(env, agent, n_transitions=200, seed=1)
    agent.global_step = agent.learning_starts + 1
    tensor_w = th.tensor([0.6, 0.4]).float().to(agent.device)
    for _ in range(5):
        agent.update(tensor_w)

    obs, _ = env.reset(seed=3)
    action = agent.eval(obs, np.array([0.6, 0.4], dtype=np.float32))
    assert 0 <= int(action) < env.action_space.n
    print(f"[ok] ECCGPIPD: 5 updates, greedy action={int(action)}")


if __name__ == "__main__":
    test_env_reward_shape()
    test_envelope_updates()
    test_gpipd_updates()
    print("\nAll smoke tests passed.")
