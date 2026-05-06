"""Training script for LexTabular on the MyFourRoom environment.

Lexicographic objective: triangle score vs circle score.
  - obj[0] = blue_triangle + red_triangle  (higher priority)
  - obj[1] = blue_circle   + red_circle    (lower  priority)

Two variants are trained back-to-back:
  1. triangle_first=True  → prioritise collecting triangles
  2. triangle_first=False → prioritise collecting circles

LexTabular is a tabular Q-learning agent that stores Q-values in a
dictionary keyed by string state representations.  Action selection is
lexicographically greedy: it prunes actions that are suboptimal for
objective 0 (within slack), then picks the best action for objective 1
among the remaining permissible actions.

The observation from MyFourRoom is already discrete:
    [row, col, collected_0, collected_1, ..., collected_15]
so string-keying works directly without any further encoding.

Usage
-----
    cd <repo_root>
    python -m run_code.lex_tabular_four_room
"""

import gymnasium as gym
import numpy as np

from env import *  # noqa: F401  – registers "my-four-room-v0"
from agent.lex_agents import LexTabular, LexTrainParams
from wrappers.lex_reward_wrapper import LexRewardWrapper

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = int(5e4)
MAX_EPISODE_STEPS = 50

EPSILON_START = 0.3
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = TOTAL_TIMESTEPS // 2

SLACK = 0.01

# Optimistic initialisation (> 0) encourages exploration in tabular methods
INIT_Q = 0.5

EVAL_EVERY = 5_000
EVAL_EPISODES = 20   # more episodes since this is fast (tabular)

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


def evaluate(agent: LexTabular, env: gym.Env, n_episodes: int) -> np.ndarray:
    """Run n_episodes with the greedy policy (epsilon=0)."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = np.zeros(2)
        while not done:
            action = agent.act(obs)
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
        f"LexTabular – priority: {priority_label} first, "
        f"then {secondary_label}\n"
        f"{'='*60}"
    )

    np.random.seed(SEED)

    train_env = make_env(triangle_first)
    eval_env = make_env(triangle_first)

    action_size = train_env.action_space.n

    params = LexTrainParams(
        reward_size=2,
        epsilon=EPSILON_START,
        slack=SLACK,
        lextab_on_policy=False,  # use Q-learning (off-policy)
    )

    agent = LexTabular(
        train_params=params,
        action_size=action_size,
        initialisation=INIT_Q,
        double=False,
    )

    global_step = 0
    obs, _ = train_env.reset(seed=SEED)

    while global_step < TOTAL_TIMESTEPS:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        agent.step(obs, action, reward, next_obs, done)

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
            n_states = len(agent.Q[0])
            print(
                f"  step {global_step:>6d} | "
                f"{obj0_name} (obj0): {mean_ret[0]:.3f} | "
                f"{obj1_name} (obj1): {mean_ret[1]:.3f} | "
                f"ε={agent.epsilon:.3f} | "
                f"#states={n_states}"
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
    print(f"Total unique states visited: {len(agent.Q[0])}")

    # Save model
    save_path = f"lex_tabular_{'tri_first' if triangle_first else 'circ_first'}"
    agent.save_model(save_path)
    print(f"Model saved to {save_path}-lex_tabular.pkl")

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
