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
from scripts.eval_envelope import eval_agent, wrapped_reward_space
from scripts.utils import find_model_path, interp_label_list, make_agent, make_env

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
    # GPI-PD takes ref_point in train() (not __init__) and has a different train()
    # signature than Envelope, so the ref_point/train wiring below branches on it.
    use_gpipd = agent_config.get("algorithm") == "gpi_pd" or agent_config.get("algorithm") == "ecc_gpi_pd"

    print( f"[train] total_timesteps={train_config.get('total_timesteps')}")
    env_ = make_env(env_config.copy())
    eval_env = make_env(env_config.copy())

    # ref_point must be strictly dominated by every Pareto solution. Derive it
    # env-agnostically from the (possibly wrapper-projected) reward space the agent
    # actually sees: scale each objective's per-step lower bound by the episode horizon
    # (worst-case undiscounted return) and subtract a margin. This works for any reward
    # dim (incl. odd dims like the 3-objective car env) and any env without reach-goal's
    # step_penalty/max_episode_steps keys, while reproducing the old reach-goal ref.
    rs = wrapped_reward_space(env_)
    horizon = env_config.get("max_episode_steps") or env_config.get("horizon") or 1
    ref_point = (np.asarray(rs.low, dtype=np.float32) * horizon - 10.0).astype(np.float32)
    # Envelope/ECC consume ref_point in __init__; GPI-PD takes it in train() instead.
    if use_gpipd:
        train_config["ref_point"] = ref_point
    else:
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
    # GPI-PD's source lives in site-packages (morl_baselines), not this repo, so there
    # is nothing under root="." to snapshot for it.
    if not use_gpipd:
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

    # Evaluate on each interpretation's projection when either:
    #  - use_ecc: the ECC policy is conditioned on a per-net weight, so it must be
    #    queried on each interpretation's projection rather than the raw reward; or
    #  - eval_both_interps: a single-policy agent was trained on one interpretation
    #    but we still want its return measured under each (e.g. how much util-help a
    #    deont-trained policy gives up).
    eval_both_interps = general_config.get("eval_both_interps", False)
    if use_ecc or eval_both_interps:
        # num_interps is ECC's per-net split (default 2 for non-ECC "full" agents).
        num_interps = getattr(agent, "num_interps", 2)
        interp_labels = interp_label_list(num_interps)
        for i, label in enumerate(interp_labels):
            # Project the raw multi-interpretation reward onto interpretation i. Pass
            # interp_index explicitly so the base env_config's choice never leaks in.
            interp_env = make_env({**env_config, "interp_index": i})
            if use_ecc and hasattr(agent, "set_eval_interp_weight"):
                # One-hot interp weight so a weight-conditioned agent (e.g. ecc_gpi_pd)
                # is queried at the interpretation it is being evaluated on.
                agent.set_eval_interp_weight(np.eye(num_interps)[i].tolist())
            # A single-policy agent is queried with its native weight dim: net_reward_dim
            # for an interpretation-trained agent (matches the projected env), or the raw
            # dim for a "full" agent evaluated on each projection. ECC keeps the default.
            weight_dim = None if use_ecc else getattr(agent, "reward_dim", None)
            eval_agent(
                interp_env, agent, out_dir, label=label, weight_dim=weight_dim,
                **eval_config, **env_config
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
