"""Training script for LexDQN on the MyFourRoom environment.

Lexicographic objective: triangle score vs circle score.
  - obj[0] = blue_triangle + red_triangle  (higher priority)
  - obj[1] = blue_circle   + red_circle    (lower  priority)

Two variants are trained back-to-back:
  1. triangle_first=True  → prioritise collecting triangles
  2. triangle_first=False → prioritise collecting circles

LexDQN is an off-policy DQN that maintains a multi-headed Q-network
(one head per objective).  At each step it selects actions that are
lexicographically optimal: it maximises objective 0, then breaks ties
by maximising objective 1 (within a small slack tolerance).

Usage
-----
    cd <repo_root>
    python -m run_code.lex_dqn_four_room
"""

import os
import sys

import gymnasium as gym
import numpy as np
import torch

from env import *  # noqa: F401  – registers "my-four-room-v0"
from agent.lex_agents import LexDQN, LexTrainParams
from wrappers.lex_reward_wrapper import LexRewardWrapper

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = int(5e4)
MAX_EPISODE_STEPS = 50

LEARNING_RATE = 1e-3
GAMMA = 0.99          # used inside LexDQN (self.discount)
HIDDEN = 64
BATCH_SIZE = 64
BUFFER_SIZE = int(2e4)
UPDATE_EVERY = 4
UPDATE_STEPS = 1
SLACK = 0.01          # permissibility tolerance for lexicographic pruning

EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = TOTAL_TIMESTEPS // 2

EVAL_EVERY = 5_000    # evaluate greedy policy every N steps
EVAL_EPISODES = 10

SEED = 42

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def make_env(triangle_first: bool) -> gym.Env:
    env = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0"),
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return LexRewardWrapper(env, triangle_first=triangle_first)


def obs_to_tensor(obs: np.ndarray) -> torch.Tensor:
    """Convert a numpy observation to a (1, obs_dim) float tensor."""
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)


def evaluate(agent: LexDQN, env: gym.Env, n_episodes: int) -> np.ndarray:
    """Run n_episodes with the greedy policy; return mean cumulative reward."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = np.zeros(agent.reward_size)
        while not done:
            state = obs_to_tensor(obs)
            action = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        totals.append(ep_reward)
    agent.epsilon = original_epsilon
    return np.mean(totals, axis=0)


def linearly_decaying_epsilon(start, end, decay_steps, current_step) -> float:
    if current_step >= decay_steps:
        return end
    return start + (end - start) * (current_step / decay_steps)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(triangle_first: bool):
    priority_label = "triangle" if triangle_first else "circle"
    secondary_label = "circle" if triangle_first else "triangle"
    print(
        f"\n{'='*60}\n"
        f"LexDQN – priority: {priority_label} first, "
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
        epsilon=EPSILON_START,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        no_cuda=True,
        update_every=UPDATE_EVERY,
        update_steps=UPDATE_STEPS,
        slack=SLACK,
        network="DNN",
        learning_rate=LEARNING_RATE,
    )

    agent = LexDQN(params, in_size=obs_size, action_size=action_size, hidden=HIDDEN)

    global_step = 0
    obs, _ = train_env.reset(seed=SEED)

    while global_step < TOTAL_TIMESTEPS:
        state = obs_to_tensor(obs)
        action = agent.act(state)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        next_state = obs_to_tensor(next_obs)
        agent.step(state, action, reward, next_state, done)

        obs = next_obs
        global_step += 1

        # Decay epsilon
        agent.epsilon = linearly_decaying_epsilon(
            EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS, global_step
        )

        if done:
            obs, _ = train_env.reset()

        # Periodic evaluation
        if global_step % EVAL_EVERY == 0:
            mean_ret = evaluate(agent, eval_env, EVAL_EPISODES)
            obj0_name = "triangle" if triangle_first else "circle"
            obj1_name = "circle" if triangle_first else "triangle"
            print(
                f"  step {global_step:>6d} | "
                f"{obj0_name} (obj0): {mean_ret[0]:.3f} | "
                f"{obj1_name} (obj1): {mean_ret[1]:.3f} | "
                f"ε={agent.epsilon:.3f}"
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

    # Save model
    save_path = f"lex_dqn_{'tri_first' if triangle_first else 'circ_first'}"
    agent.save_model(save_path)
    print(f"Model saved to {save_path}-lex_dqn.pt")

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
