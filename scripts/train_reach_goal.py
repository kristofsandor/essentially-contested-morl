"""Train an Envelope agent on the Specified env and visualize the Pareto front.

- ``train``: train an Envelope agent (weight-conditioned, covers the full
  simplex in one run), evaluate it at a weight sweep, and save a 2-D Pareto
  front plot.  The trade-off is task efficiency (minimise steps to goal)
  vs. how many humans are helped along the way.

Run from the repo root::

    python -m scripts.train_reach_goal --mode train --total-timesteps 100000
"""

from __future__ import annotations

import wandb
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.pareto import get_non_pareto_dominated_inds
from morl_baselines.multi_policy.envelope.envelope import Envelope
import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import env  # noqa: F401, registers the envs on import

COMPONENT_NAMES = ["task", "help"]


def make_env(env_config) -> gym.Env:
    env = gym.make(**env_config)
    return env


# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def run_train(config) -> None:

    env_config = config["env"]
    agent_config = config["agent"]
    train_config = config["train"]
    eval_config = config["eval"]

    load_agent_path = agent_config.pop("load_agent_path", "")


    print(
        f"[train] total_timesteps={train_config.get('total_timesteps')}, "
        f"num_eval_weights={train_config.get('num_eval_weights')}, num_eval_episodes={train_config.get('num_eval_episodes')}"
    )
    env_ = make_env(env_config)
    eval_env = make_env(env_config)

    # task return ∈ [−step_penalty*max_steps, 0], help return ∈ [0, num_humans*help_reward].
    # ref_point must be strictly dominated by every Pareto solution.
    ref_point = np.array(
        [
            -(
                env_config.get("step_penalty") * env_config.get("max_episode_steps")
                + 10.0
            ),
            -0.1,
        ],
        dtype=np.float32,
    )

    # Smaller network and higher LR than fire_rescue: the obs is 84-D and
    # the Pareto front is 1-D (2 objectives), so learning is much faster.
    agent = Envelope(
        env=env_,
        **agent_config,
    )
    if load_agent_path:
        print(f"[train] loading agent from {load_agent_path} ...")
        agent.load(load_agent_path)

    wandb.log(config)

    print("[train] training Envelope ...")
    agent.train(eval_env=eval_env, ref_point=ref_point, **train_config)

    use_small = env_config.get("grid_size") <= 5
    run_id = wandb.run.id
    run_id = f"{"small" if use_small else "medium"}__{train_config.get('total_timesteps')}__{run_id}"

    min_x = - env_config.get("max_episode_steps") * env_config.get("step_penalty") - 10.0
    eval(eval_env, agent, run_id, min_x, **eval_config)


def eval(eval_env, agent, run_id, min_x, num_eval_weights, num_eval_episodes, out_dir) -> None:
    print("[train] evaluating at a sweep of weights ...")
    out_dir = Path(out_dir)
    out_dir = out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent.save(save_dir=str(out_dir), filename="model")
    except MemoryError as e:
        print(f"[train] error saving agent: {e}, retrying without replay buffer")
    try:
        agent.save(save_dir=str(out_dir), filename="model", save_replay_buffer=False)
    except Exception as e:
        print(f"[train] error saving agent: {e}")

    reward_dim = eval_env.unwrapped.reward_space.shape[0]
    weights = equally_spaced_weights(reward_dim, n=num_eval_weights)
    eval_returns = []
    for w in weights:
        ep_returns = []
        for ep in range(num_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ret = np.zeros(reward_dim, dtype=np.float32)
            while not done:
                action = agent.eval(obs, np.asarray(w, dtype=np.float32))
                obs, reward, terminated, truncated, _ = eval_env.step(int(action))
                ret += reward
                done = terminated or truncated
            ep_returns.append(ret)
        eval_returns.append(np.mean(ep_returns, axis=0))
    eval_returns = np.stack(eval_returns)  # (num_weights, 2)


    np.save(out_dir / "eval_returns.npy", eval_returns)
    np.save(out_dir / "eval_weights.npy", np.asarray(weights, dtype=np.float32))
    print(f"[train] saved raw returns to {out_dir / 'eval_returns.npy'}")

    plot_eval(eval_returns, weights, out_dir, name="eval_returns", min_x=min_x)

    nd_idxs = get_non_pareto_dominated_inds(eval_returns, remove_duplicates=True)
    pf = eval_returns[nd_idxs]
    pf_weights = np.asarray(weights, dtype=np.float32)[nd_idxs]

    plot_eval(pf, pf_weights, out_dir, name="pareto_front", min_x=min_x)

def plot_eval(data, weights, out_dir, name, min_x):
    """
    Logs the table and plots on wandb, and also saves the plot locally. Called from eval mode.
    
    """
    # log wandb table
    table = wandb.Table(data=data, columns=COMPONENT_NAMES)
    wandb.log(
        {
            name: table,
            f"{name}_weights": wandb.Table(
                data=weights, columns=[f"{c}_weight" for c in COMPONENT_NAMES]
            ),
        }
    )

    wandb.log({
        f"{name}_scatter": wandb.plot.scatter(
            table,
            x=COMPONENT_NAMES[0],
            y=COMPONENT_NAMES[1],
            title=f"{name}: task efficiency vs. humans helped",
    )})

    # save the plots to wandb
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(data[:, 0], data[:, 1], s=30, zorder=3)
    for w_idx, w in enumerate(weights):
        ax.annotate(
            f"({w[0]:.2f},{w[1]:.2f})",
            (data[w_idx, 0], data[w_idx, 1]),
            fontsize=7,
            alpha=0.75,
        )
    ax.set_xlabel(COMPONENT_NAMES[0])
    ax.set_ylabel(COMPONENT_NAMES[1])
    ax.set_title(f"{name}: task efficiency vs. humans helped")
    ax.set_xlim((min_x, data[:, 0].max() + 2))
    ax.set_ylim((-0.2, 5 + 2))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=140)
    wandb.log({f"{name}_plot": wandb.Image(str(path))})
    plt.close(fig)

    print(f"[train] saved {name} plot to {path}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON file containing the configuration for the environment.",
    )
    args = parser.parse_args()
    config = json.load(open(args.config))
    run_train(config)


if __name__ == "__main__":
    main()
