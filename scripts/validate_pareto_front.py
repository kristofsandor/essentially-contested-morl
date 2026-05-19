"""Validate that the cut-down FireRescue env has a non-degenerate Pareto front.

Two modes:

- ``smoke`` (default, seconds): roll out random policies for a handful of
  episodes, print per-component reward statistics, confirm the obs/reward
  shapes after wrapping, and check that all reward dimensions actually move
  (a constant-zero dimension would be a giveaway that something is broken).

- ``train``: train an Envelope agent (which is internally weight-conditioned,
  so a single training run covers the whole simplex), then evaluate it at a
  sweep of evenly spaced weights and save 2-D projection plots of the
  resulting return vectors. A non-degenerate Pareto front shows up as a
  visible spread along each pair of axes; a flat blob means the methodology
  isn't separating the interpretations.

Run from the repo root::

    python -m scripts.validate_pareto_front --mode smoke
    python -m scripts.validate_pareto_front --mode train --total-timesteps 50000

The script keeps hyperparameters modest so it can finish on a CPU laptop;
crank ``--total-timesteps`` and ``--num-eval-weights`` once you have access
to a GPU or a longer compute budget.
"""

from __future__ import annotations
import wandb

import argparse
import itertools
import os
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # safe to import on headless machines
import matplotlib.pyplot as plt
import numpy as np

# The env package registers FireRescue-v0 on import.
import env  # noqa: F401
from env.fire_rescue import FireRescueEnv
from env.wrappers import FlattenFireRescueObs


# Reward component names in the order the cut-down env emits them.
COMPONENT_NAMES_4D = [
    "task",
    "safety_sentient",
    "safety_classical",
    "safety_hedonistic",
]
COMPONENT_NAMES_7D = COMPONENT_NAMES_4D + [
    "fairness_equal",
    "fairness_proportional",
    "fairness_minimum",
]


def make_env(
    include_fairness: bool,
    seed: int,
    small: bool = False,
) -> gym.Env:
    """Build the cut-down env with the obs-flatten wrapper.

    If ``small=True``, use a much smaller config (5x5 grid, 2 humans + 1 dog,
    2 diamonds, 50 max steps, 1 initial fire cell, fire_spread_prob=0.2).
    Use this for fast iteration / debugging the methodology — successful
    rescues happen ~10x more often per episode than in the default config,
    so Envelope sees enough positive signal to learn within tens of
    thousands of steps instead of millions.
    """
    if small:
        base = FireRescueEnv(
            size=5,
            num_humans=8,
            num_dogs=8,
            num_diamonds=2,
            max_steps=50,
            initial_fire_cells=1,
            fire_spread_prob=0.2,
            include_fairness=include_fairness,
            enable_self_rescue=False,
            use_delta_rewards=True,
            normalize_rewards=True,
        )
    else:
        base = FireRescueEnv(
            include_fairness=include_fairness,
            enable_self_rescue=False,
            use_delta_rewards=True,
            normalize_rewards=True,
        )
    env_ = FlattenFireRescueObs(base)
    env_.reset(seed=seed)
    return env_


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------


def run_smoke(num_episodes: int, include_fairness: bool, seed: int) -> None:
    print(f"[smoke] include_fairness={include_fairness}, episodes={num_episodes}")
    env_ = make_env(include_fairness, seed)
    print(f"[smoke] obs_space  = {env_.observation_space}")
    print(f"[smoke] act_space  = {env_.action_space}")
    print(f"[smoke] reward_dim = {env_.unwrapped.reward_dim}")
    names = COMPONENT_NAMES_7D if include_fairness else COMPONENT_NAMES_4D

    returns = []
    for ep in range(num_episodes):
        obs, _ = env_.reset(seed=seed + ep)
        assert obs.shape == env_.observation_space.shape, (
            f"obs shape {obs.shape} != space {env_.observation_space.shape}"
        )
        ep_return = np.zeros(env_.unwrapped.reward_dim, dtype=np.float32)
        terminated = truncated = False
        while not (terminated or truncated):
            action = env_.action_space.sample()
            obs, reward, terminated, truncated, _ = env_.step(action)
            ep_return += reward
        returns.append(ep_return)

    returns = np.stack(returns)
    print("\n[smoke] per-component return statistics over random rollouts:")
    print(
        f"  {'component':<25s} {'min':>10s} {'mean':>10s} {'max':>10s} {'std':>10s}"
    )
    for i, name in enumerate(names):
        col = returns[:, i]
        print(
            f"  {name:<25s} {col.min():>10.3f} {col.mean():>10.3f} "
            f"{col.max():>10.3f} {col.std():>10.3f}"
        )

    # Sanity: warn about constant-zero dimensions.
    constant = [names[i] for i in range(returns.shape[1]) if returns[:, i].std() < 1e-8]
    if constant:
        print(
            f"\n[smoke] WARNING: these components never moved across "
            f"{num_episodes} random rollouts: {constant}"
        )
        print(
            "        Either the dynamic that drives them is too rare, or the "
            "reward emission is broken."
        )
    else:
        print("\n[smoke] all components varied across rollouts (good).")


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def run_train(
    total_timesteps: int,
    num_eval_weights: int,
    num_eval_episodes: int,
    include_fairness: bool,
    seed: int,
    out_dir: Path,
) -> None:
    # Lazy import so the smoke mode does not need the full morl_baselines stack.
    from morl_baselines.common.weights import equally_spaced_weights
    from morl_baselines.multi_policy.envelope.envelope import Envelope

    print(
        f"[train] include_fairness={include_fairness}, "
        f"total_timesteps={total_timesteps}, num_eval_weights={num_eval_weights}"
    )
    env_ = make_env(include_fairness, seed, small=True)  # small config for faster training
    eval_env = make_env(include_fairness, seed + 10_000, small=True)

    reward_dim = env_.unwrapped.reward_dim
    names = COMPONENT_NAMES_7D if include_fairness else COMPONENT_NAMES_4D

    # The reward components are roughly normalised to [0, 1], so a slightly
    # negative ref point is a safe choice for hypervolume.
    ref_point = -0.1 * np.ones(reward_dim, dtype=np.float32)

    agent = Envelope(
        env=env_,
        learning_rate=3e-4,
        net_arch=[256, 256],
        batch_size=128,
        learning_starts=1000,
        gradient_updates=1,
        target_net_update_freq=200,
        log=True,
    )

    run_name = wandb.run.id
    out_dir = out_dir / run_name

    print("[train] training Envelope with random weight sampling ...")
    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=ref_point,
        weight=None,  # weight-conditioned: sampled per episode internally
        eval_freq=max(1000, total_timesteps // 5),
        num_eval_weights_for_front=num_eval_weights,
        num_eval_episodes_for_front=num_eval_episodes,
    )

    agent.save(save_dir=str(out_dir), filename="model")


    print("[train] evaluating at a sweep of weights ...")
    weights = equally_spaced_weights(reward_dim, n=num_eval_weights)
    eval_returns = []
    for w in weights:
        ep_returns = []
        for ep in range(num_eval_episodes):
            obs, _ = eval_env.reset(seed=seed + 50_000 + ep)
            done = False
            ret = np.zeros(reward_dim, dtype=np.float32)
            while not done:
                action = agent.eval(obs, np.asarray(w, dtype=np.float32))
                obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                ret += reward
                done = terminated or truncated
            ep_returns.append(ret)
        eval_returns.append(np.mean(ep_returns, axis=0))
    eval_returns = np.stack(eval_returns)  # (num_weights, reward_dim)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eval_returns.npy", eval_returns)
    np.save(out_dir / "eval_weights.npy", np.asarray(weights, dtype=np.float32))
    print(f"[train] saved raw returns to {out_dir / 'eval_returns.npy'}")

    # 2-D projection plots for every pair of components.
    pairs = list(itertools.combinations(range(reward_dim), 2))
    for i, j in pairs:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.scatter(eval_returns[:, i], eval_returns[:, j], s=18)
        for w_idx, w in enumerate(weights):
            ax.annotate(
                f"({w[i]:.2f},{w[j]:.2f})",
                (eval_returns[w_idx, i], eval_returns[w_idx, j]),
                fontsize=6,
                alpha=0.7,
            )
        ax.set_xlabel(names[i])
        ax.set_ylabel(names[j])
        ax.set_title(f"{names[i]} vs {names[j]}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"front_{names[i]}_vs_{names[j]}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"[train] wrote {path}")

    print(
        "\n[train] Inspect the projection plots: a non-degenerate Pareto front "
        "shows visible spread along each axis pair. A clustered blob means "
        "the env or the algorithm is failing to separate interpretations."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-fairness",
        action="store_true",
        help="Use the full 7-dim reward (default: 4-dim cut-down).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=20, help="Smoke mode only."
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=50_000, help="Train mode only."
    )
    parser.add_argument(
        "--num-eval-weights", type=int, default=15, help="Train mode only."
    )
    parser.add_argument(
        "--num-eval-episodes", type=int, default=5, help="Train mode only."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/fire_rescue/pareto_front_validation",
        help="Train mode only. Where to save plots and arrays.",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        run_smoke(
            num_episodes=args.num_episodes,
            include_fairness=args.include_fairness,
            seed=args.seed,
        )
    else:
        run_train(
            total_timesteps=args.total_timesteps,
            num_eval_weights=args.num_eval_weights,
            num_eval_episodes=args.num_eval_episodes,
            include_fairness=args.include_fairness,
            seed=args.seed,
            out_dir=Path(args.out_dir),
        )


if __name__ == "__main__":
    main()
