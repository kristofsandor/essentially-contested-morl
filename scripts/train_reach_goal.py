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
from pathlib import Path

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

    general_config = config.get("general", {})
    env_config = config["env"]
    agent_config = config["agent"]
    train_config = config["train"]
    eval_config = config["eval"]

    run_id = general_config.get("continue_run_id", "")
    # ECC trains per-interpretation nets on a 2-objective [task, help] weight, so it
    # must be evaluated on each interpretation's projection rather than the raw reward.
    use_ecc = general_config.get("use_ecc", agent_config.get("algorithm") == "ecc_envelope")

    print( f"[train] total_timesteps={train_config.get('total_timesteps')}")
    env_ = make_env(env_config.copy())
    eval_env = make_env(env_config.copy())

    # task return ∈ [−step_penalty*max_steps, 0], help return ∈ [0, num_humans*help_reward].
    # ref_point must be strictly dominated by every Pareto solution.
    ref_point = np.array(
        [
            -env_config.get("step_penalty") * env_config.get("max_episode_steps") - 10,
            -0.1,
        ],
        dtype=np.float32,
    )
    agent_config["ref_point"] = ref_point

    agent = make_agent(env=env_, agent_config=agent_config)

    if run_id:
        agent_path = find_model_path(run_id)
        print(f"[train] loading agent from {agent_path} ...")
        agent.load(agent_path)

    agent.register_additional_config(
        {
            "general": general_config,
            "train": train_config,
            "env": env_config,
            "agent": agent_config,
            "eval": eval_config,
        }
    )

    # Snapshot the agent implementation to W&B's Code tab for this run, so the run
    # records exactly which learner source produced it.
    if getattr(agent, "log", False) and wandb.run is not None:
        agent_src = "agent/ecc_envelope.py" if use_ecc else "agent/envelope.py"
        wandb.run.log_code(
            root=".",
            include_fn=lambda path, *_: path.replace("\\", "/").endswith(agent_src),
        )

    print("[train] training Envelope ...")
    agent.train(eval_env=eval_env, **train_config)

    run_id = f"small__{train_config.get('total_timesteps')}__{wandb.run.id}"

    out_dir = train_config.get("out_dir", "results/reach_goal/pareto_front")
    out_dir = Path(out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent.save(save_dir=str(out_dir), filename="model")
    except MemoryError as e:
        print(f"[train] error saving agent: {e}, retrying without replay buffer")
    try:
        agent.save(save_dir=str(out_dir), filename="model", save_replay_buffer=False)
    except Exception as e:
        print(f"[train] error saving agent: {e}")

    # eval_config may carry its own out_dir; the run's out_dir (above) wins.
    eval_config.pop("out_dir", None)

    if use_ecc:
        # The ECC policy is conditioned on a 2-objective [task, help] weight, so it
        # is evaluated on each ethical interpretation's 2-objective projection of the
        # reward rather than on the raw multi-interpretation env reward.
        for interp in ("deontological", "utilitarian"):
            interp_env = make_env({**env_config, f"{interp}_wrapper": True})
            eval_agent(
                interp_env, agent, out_dir, label=interp, **eval_config, **env_config
            )
    else:
        eval_agent(eval_env, agent, out_dir, **eval_config, **env_config)


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
