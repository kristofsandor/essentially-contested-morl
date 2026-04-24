"""ECC-MORLD on MyFourRoom.

Uses MORLD (a multi-policy MORL algorithm) to find a Pareto front *between* two
contested concepts while applying *strict lexicographic ordering within* each
concept's interpretations.

Concept definitions for MyFourRoom
-----------------------------------
The environment emits a 4-D reward vector:
    index 0 — blue_triangle   ┐ Concept A (e.g. "colour-blue" interpretation)
    index 1 — blue_circle     ┘   priority: blue_triangle > blue_circle
    index 2 — red_triangle    ┐ Concept B (e.g. "colour-red" interpretation)
    index 3 — red_circle      ┘   priority: red_triangle > red_circle

After wrapping with LexRewardWrapper the agent receives a 2-D vector:
    [lex_blue, lex_red]

where:
    lex_blue = blue_triangle * M  +  blue_circle
    lex_red  = red_triangle  * M  +  red_circle

and M = MAX_EPISODE_LENGTH + 1 guarantees strict ordering for binary rewards.

MORLD then produces a Pareto front over [lex_blue, lex_red], representing policies
that trade off between the two contested concepts in a Pareto-optimal way, while
within each concept the lexicographic priority is always respected.
"""

import gymnasium as gym
import numpy as np
from morl_baselines.multi_policy.morld.morld import MORLD

import env  # noqa: F401 — registers "my-four-room-v0"
from utils.config import EXPERIMENT_NAME, MAX_EPISODE_LENGTH, TOTAL_TIMESTEPS
from utils.visualize_front import visualize_front_general
from wrappers import LexRewardWrapper

# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------
# Each sub-list is one contested concept; indices refer to the original 4-D
# reward vector.  Items are listed in *descending priority* order.
CONCEPT_GROUPS = [
    [0, 1],  # Concept A: [blue_triangle (high), blue_circle (low)]
    [2, 3],  # Concept B: [red_triangle (high), red_circle  (low)]
]

# Big-M scale: must exceed the maximum possible undiscounted sum of any single
# interpretation over one episode.  For binary rewards over MAX_EPISODE_LENGTH
# steps, M = MAX_EPISODE_LENGTH + 1 is a tight and provably sufficient choice.
LEX_SCALE = MAX_EPISODE_LENGTH + 1

# Reference point for the 2-D concept space used by MORLD's hypervolume metric.
# Both dimensions are in the lex-scalarized space, so the worst possible value is
# 0 (no blue/red items collected at all).
REF_POINT = np.array([0.0, 0.0])


def make_env() -> gym.Env:
    """Build the wrapped environment."""
    base = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0"),
        max_episode_steps=MAX_EPISODE_LENGTH,
    )
    return LexRewardWrapper(
        base,
        concept_groups=CONCEPT_GROUPS,
        lex_scale=LEX_SCALE,
    )


def main() -> None:
    env_train = make_env()
    env_eval = make_env()

    agent = MORLD(
        env=env_train,
        scalarization_method="ws",
        evaluation_mode="ser",
        policy_name="MOSACDiscrete",
        experiment_name=f"ECC_MORLD_lex_{EXPERIMENT_NAME}",
    )

    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=env_eval,
        ref_point=REF_POINT,
    )

    visualize_front_general(
        agent.archive.evaluations,
        columns=["lex_blue", "lex_red"],
    )

    agent.close_wandb()


if __name__ == "__main__":
    main()
