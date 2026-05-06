"""Training script for LexActorCritic on the MyFourRoom environment.

Lexicographic objective: triangle score vs circle score.
  - obj[0] = blue_triangle + red_triangle  (higher priority)
  - obj[1] = blue_circle   + red_circle    (lower  priority)

Two variants are trained back-to-back:
  1. triangle_first=True  → prioritise collecting triangles
  2. triangle_first=False → prioritise collecting circles

LexActorCritic is an on-policy actor-critic (A2C or PPO) that uses
Lagrange multipliers to enforce a soft lexicographic constraint: it
maximises a weighted combination of objectives where the weights are
automatically adjusted so that higher-priority objectives are never
sacrificed beyond a small threshold (controlled by the Lagrange mu).

Usage
-----
    cd <repo_root>
    python -m run_code.lex_actor_critic_four_room
"""

import gymnasium as gym
import numpy as np
import torch

from env import *  # noqa: F401  – registers "my-four-room-v0"
from agent.lex_agents import LexActorCritic, LexTrainParams
from wrappers.lex_reward_wrapper import LexRewardWrapper

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = int(5e4)
MAX_EPISODE_STEPS = 50

LEARNING_RATE = 3e-4   # critic LR; actor LR = 0.01 × this (see LexActorCritic)
HIDDEN = 64
BATCH_SIZE = 128       # on-policy batch: update every BATCH_SIZE steps
BUFFER_SIZE = BATCH_SIZE  # only the current on-policy batch is kept
SLACK = 0.01

MODE = "a2c"           # "a2c" or "ppo"
SEQUENTIAL = False     # update all objectives jointly using Lagrange multipliers

EVAL_EVERY = 5_000
EVAL_EPISODES = 10

SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(triangle_first: bool) -> gym.Env:
    env = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0"),
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return LexRewardWrapper(env, triangle_first=triangle_first)


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def evaluate(agent: LexActorCritic, env: gym.Env, n_episodes: int) -> np.ndarray:
    """Run n_episodes with the greedy (max-probability) policy."""
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = np.zeros(2)
        while not done:
            state = obs_to_tensor(obs)
            with torch.no_grad():
                probs = agent.actor(state)
            action = probs.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        totals.append(ep_reward)
    return np.mean(totals, axis=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(triangle_first: bool):
    priority_label = "triangle" if triangle_first else "circle"
    secondary_label = "circle" if triangle_first else "triangle"
    print(
        f"\n{'='*60}\n"
        f"LexActorCritic ({MODE.upper()}) – priority: {priority_label} first, "
        f"then {secondary_label}\n"
        f"{'='*60}"
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_env = make_env(triangle_first)
    eval_env = make_env(triangle_first)

    obs_size = train_env.observation_space.shape[0]
    action_size = train_env.action_space.n

    params = LexTrainParams(
        reward_size=2,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        slack=SLACK,
        learning_rate=LEARNING_RATE,
        no_cuda=True,
    )

    agent = LexActorCritic(
        train_params=params,
        in_size=obs_size,
        action_size=action_size,
        mode=MODE,
        second_order=False,
        sequential=SEQUENTIAL,
        hidden=HIDDEN,
    )

    global_step = 0
    obs, _ = train_env.reset(seed=SEED)

    while global_step < TOTAL_TIMESTEPS:
        state = obs_to_tensor(obs)
        action = agent.act(state)
        next_obs, reward, terminated, truncated, _ = train_env.step(action.item())
        done = terminated or truncated

        next_state = obs_to_tensor(next_obs)
        agent.step(state, action, reward, next_state, float(done))

        obs = next_obs
        global_step += 1

        if done:
            obs, _ = train_env.reset()

        # Periodic evaluation
        if global_step % EVAL_EVERY == 0:
            mean_ret = evaluate(agent, eval_env, EVAL_EPISODES)
            obj0_name = "triangle" if triangle_first else "circle"
            obj1_name = "circle" if triangle_first else "triangle"
            mu_str = ", ".join(f"{m:.4f}" for m in agent.mu)
            print(
                f"  step {global_step:>6d} | "
                f"{obj0_name} (obj0): {mean_ret[0]:.3f} | "
                f"{obj1_name} (obj1): {mean_ret[1]:.3f} | "
                f"mu={mu_str}"
            )

    # Final evaluation
    mean_ret = evaluate(agent, eval_env, EVAL_EPISODES * 2)
    obj0_name = "triangle" if triangle_first else "circle"
    obj1_name = "circle" if triangle_first else "triangle"
    print(
        f"\nFinal ({priority_label} first) – "
        f"{obj0_name}: {mean_ret[0]:.3f}, "
        f"{obj1_name}: {mean_ret[1]:.3f}"
    )
    print(f"Final Lagrange mu: {agent.mu}")

    # Save model
    save_path = f"lex_ac_{MODE}_{'tri_first' if triangle_first else 'circ_first'}"
    agent.save_model(save_path)
    print(f"Model saved ({save_path}-lex_ac_{{actor,critic}}.pt)")

    train_env.close()
    eval_env.close()

    return mean_ret


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ret_tri_first = train(triangle_first=True)
    ret_circ_first = train(triangle_first=False)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(
        f"Triangle-first → triangle: {ret_tri_first[0]:.3f}, "
        f"circle: {ret_tri_first[1]:.3f}"
    )
    print(
        f"Circle-first   → circle:   {ret_circ_first[0]:.3f}, "
        f"triangle: {ret_circ_first[1]:.3f}"
    )
