import argparse
import json
import json
from pathlib import Path
import gymnasium as gym
import wandb
import numpy as np
import matplotlib.pyplot as plt
import env  # noqa: F401, registers the envs on import

from scripts.utils import find_model_path, make_agent
from morl_baselines.common.pareto import get_non_pareto_dominated_inds
from morl_baselines.common.weights import equally_spaced_weights

COMPONENT_NAMES = ["task", "help"]


def wrapped_reward_space(env):
    """Return the outermost ``reward_space`` in the wrapper chain.

    ``env.unwrapped.reward_space`` skips reward wrappers (e.g. the per-interpretation
    projections), so it reports the raw env reward dim. Here we walk outward and take
    the first wrapper that defines ``reward_space`` — i.e. the reward space the agent
    actually sees — falling back to the base env's.
    """
    e = env
    while e is not None:
        rs = e.__dict__.get("reward_space")
        if rs is not None:
            return rs
        e = getattr(e, "env", None)
    return env.unwrapped.reward_space

def eval_agent(
    eval_env,
    agent,
    out_dir,
    num_eval_weights,
    num_eval_episodes,
    max_episode_steps,
    num_humans,
    help_reward,
    step_penalty,
    terminal_reward,
    proximity_reward,
    label="",
    use_ecc=False,
    **__unused_kwargs,
) -> None:
    if use_ecc:
        ecc_weights = equally_spaced_weights(2, n=5)  # only for ECC, ignored by others
        for ecc_weight in ecc_weights:
            suffix = f"_{label}_{ecc_weight[0]}_{ecc_weight[1]}"
            eval_agent_(
                eval_env,
                agent,
                out_dir,
                num_eval_weights,
                num_eval_episodes,
                max_episode_steps,
                num_humans,
                help_reward,
                step_penalty,
                terminal_reward,
                proximity_reward,
                label=suffix,
                ecc_weight=ecc_weight,
            )
    else:
        suffix = f"_{label}" if label else ""
        eval_agent_(
            eval_env,
            agent,
            out_dir,
            num_eval_weights,
            num_eval_episodes,
            max_episode_steps,
            num_humans,
            help_reward,
            step_penalty,
            terminal_reward,
            proximity_reward,
            label=suffix,
            ecc_weight=None,
        )

def eval_agent_(
    eval_env,
    agent,
    out_dir,
    num_eval_weights,
    num_eval_episodes,
    max_episode_steps,
    num_humans,
    help_reward,
    step_penalty,
    terminal_reward,
    proximity_reward,
    label="",
    ecc_weight=None,
    **__unused_kwargs,
) -> None:
    print(f"[train] evaluating at a sweep of weights{f' ({label})' if label else ''} ...")
    reward_dim = wrapped_reward_space(eval_env).shape[0]
    weights = equally_spaced_weights(reward_dim, n=num_eval_weights)
    eval_returns = []
    for w in weights:
        ep_returns = _eval(eval_env, agent, w, num_eval_episodes, reward_dim, interp_w=ecc_weight)
        eval_returns.append(np.mean(ep_returns, axis=0))
    eval_returns = np.stack(eval_returns)  # (num_weights, 2)

    np.save(out_dir / f"eval_returns{label}.npy", eval_returns)
    np.save(out_dir / f"eval_weights{label}.npy", np.asarray(weights, dtype=np.float32))
    print(f"[train] saved raw returns to {out_dir / f'eval_returns{label}.npy'}")

    plot_eval(
        data=eval_returns,
        weights=weights,
        out_dir=out_dir,
        name=f"pareto_front{label}",
        max_episode_steps=max_episode_steps,
        num_humans=num_humans,
        help_reward=help_reward,
        step_penalty=step_penalty,
        terminal_reward=terminal_reward,
        proximity_reward=proximity_reward,
    )

    # nd_idxs = get_non_pareto_dominated_inds(eval_returns, remove_duplicates=True)
    # pf = eval_returns[nd_idxs]
    # pf_weights = np.asarray(weights, dtype=np.float32)[nd_idxs]

    # plot_eval(
    #     data=pf,
    #     weights=pf_weights,
    #     out_dir=out_dir,
    #     name="pareto_front",
    #     max_episode_steps=max_episode_steps,
    #     num_humans=num_humans,
    #     help_reward=help_reward,
    #     step_penalty=step_penalty,
    #     terminal_reward=terminal_reward,
    #     proximity_reward=proximity_reward,
    # )

def _eval(eval_env, agent, w,  num_eval_episodes, reward_dim, interp_w = None):
        ep_returns = []
        for ep in range(num_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ret = np.zeros(reward_dim, dtype=np.float32)
            while not done:
                if interp_w is not None:
                    action = agent.eval(obs, np.asarray(w, dtype=np.float32), interp_w)
                else:
                    action = agent.eval(obs, np.asarray(w, dtype=np.float32))
                obs, reward, terminated, truncated, _ = eval_env.step(int(action))

                ret += reward
                done = terminated or truncated
            ep_returns.append(ret)
        return ep_returns


def plot_eval(
    data,
    weights,
    out_dir,
    name,
    max_episode_steps,
    num_humans,
    help_reward,
    step_penalty,
    terminal_reward,
    proximity_reward,
):
    """
    Logs the table and plots on wandb, and also saves the plot locally. Called from eval mode.

    """
    nd_idxs = get_non_pareto_dominated_inds(data, remove_duplicates=True)
    pf = data[nd_idxs]
    pf_weights = np.asarray(weights, dtype=np.float32)[nd_idxs]

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

    wandb.log(
        {
            f"{name}_scatter": wandb.plot.scatter(
                table,
                x=COMPONENT_NAMES[0],
                y=COMPONENT_NAMES[1],
                title=f"{name}: task efficiency vs. humans helped",
            )
        }
    )

    # save the plots to wandb
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(data[:, 0], data[:, 1], s=30, zorder=3, color="steelblue", label="evaluated")
    ax.scatter(pf[:, 0], pf[:, 1], s=50, zorder=4, color="red", label="pareto front")
    ax.legend(fontsize=8)
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
    ax.grid(True, alpha=0.3)

    x_min = -max_episode_steps * step_penalty - 0.2
    x_max = terminal_reward + 0.2
    y_min = -max_episode_steps * proximity_reward
    y_max = num_humans * help_reward + max_episode_steps * proximity_reward + 0.2
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=140)
    wandb.log({f"{name}_plot": wandb.Image(str(path))})
    plt.close(fig)

    print(f"[train] saved {name} plot to {path}")


# def get_config_wandb(run_id):
#     """
#     Retrieves the whole config for a given run_id from wandb, including env_config, agent_config, eval_config and train_config.

#     """
#     # load config from summary metrics of the run (this is where we log the whole config as a dict in train_reach_goal.py)
#     api = wandb.Api()
#     run = api.run(f"MORL-Baselines/{run_id}")
#     config = run.config
#     return config


def eval_run_id(run_id, config=None, render=False) -> None:
    print(f"[eval] loading config from wandb for run {run_id} ...")
    wandb.init(id=run_id, resume="allow", project="MORL-Baselines")


    if config is None:
        config = wandb.run.config

    env_config = config["env"]
    agent_config = config["agent"]
    eval_config = config["eval"]

    if render:
        env_config["render_mode"] = "human"
    eval_env = gym.make(**env_config)

    agent_path = find_model_path(run_id)
    agent = make_agent(env=eval_env, agent_config=agent_config)
    agent.load(agent_path)

    eval_agent(eval_env, agent, run_id, **eval_config, **env_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON file containing the configuration for the environment.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="ID of the run to evaluate.",
    )
    # if -- render flag is there, render it, otherwise don't render
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the evaluation.",
    )
    args = parser.parse_args()

    if args.config is not None:
        config = json.load(open(args.config))
    else:
        config = None

    eval_run_id(args.run_id, config, render=args.render)


if __name__ == "__main__":
    main()
