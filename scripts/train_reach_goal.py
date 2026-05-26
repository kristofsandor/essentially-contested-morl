"""Train an Envelope agent on the Specified env and visualize the Pareto front.

- ``train``: train an Envelope agent (weight-conditioned, covers the full
  simplex in one run), evaluate it at a weight sweep, and save a 2-D Pareto
  front plot.  The trade-off is task efficiency (minimise steps to goal)
  vs. how many humans are helped along the way.

Run from the repo root::

    python -m scripts.train_reach_goal --mode train --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import json

import matplotlib

import wandb
from scripts.eval_envelope import eval_agent
from scripts.utils import find_model_path, make_agent, make_env

matplotlib.use("Agg")
import numpy as np

import env  # noqa: F401, registers the envs on import

# ---------------------------------------------------------------------------
# Train mode
# ---------------------------------------------------------------------------


def run_train(config) -> None:

    env_config = config["env"]
    agent_config = config["agent"]
    train_config = config["train"]
    eval_config = config["eval"]

    run_id = agent_config.pop("continue_run_id", "")

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
            -env_config.get("step_penalty") * env_config.get("max_episode_steps") - 10,
            -0.1,
        ],
        dtype=np.float32,
    )

    agent = make_agent(env=env_, agent_config=agent_config)

    if run_id:
        agent_path = find_model_path(run_id)
        print(f"[train] loading agent from {agent_path} ...")
        agent.load(agent_path)

    agent.register_additional_config(
        {
            "train": train_config,
            "env": env_config,
            "agent": agent_config,
            "eval": eval_config,
        }
    )

    print("[train] training Envelope ...")
    agent.train(eval_env=eval_env, ref_point=ref_point, **train_config)

    use_small = env_config.get("grid_size") <= 5
    run_id = wandb.run.id
    run_id = f"{"small" if use_small else "medium"}__{train_config.get('total_timesteps')}__{run_id}"

    min_x = -env_config.get("max_episode_steps") * env_config.get("step_penalty") - 10.0
    eval_agent(eval_env, agent, run_id, min_x, **eval_config, **env_config)


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
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        with wandb.init():
            config = {
                "env": dict(wandb.config["env"]),
                "agent": dict(wandb.config["agent"]),
                "train": dict(wandb.config["train"]),
                "eval": dict(wandb.config["eval"]),
            }
            run_train(config)
    else:
        config = json.load(open(args.config))
        run_train(config)


if __name__ == "__main__":
    main()
