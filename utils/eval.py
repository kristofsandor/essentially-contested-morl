from typing import Optional, List
import gymnasium as gym
from morl_baselines.common.evaluation import (
    cardinality,
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
)
from morl_baselines.common.pareto import ParetoArchive, filter_pareto_dominated
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.common.weights import equally_spaced_weights
import numpy as np

from morl_baselines.multi_policy.morld.morld import Policy
from pymoo.util.ref_dirs import get_reference_directions
import wandb

from utils.config import MAX_EPISODE_LENGTH
from utils.visualize_front import visualize_front_general


def eval_full_four_room(agent):
    test_env = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0"), max_episode_steps=MAX_EPISODE_LENGTH
    )
    evals, archive = eval_all_policies(
        population=agent.population, eval_env=test_env, ref_point=np.array([0, 0, 0, 0])
    )

    visualize_front_general(
        archive.evaluations,
        columns=["blue_triangle", "blue_circle", "red_triangle", "red_circle"],
    )


def eval_policy(
    policy: Policy,
    eval_env: gym.Env,
    num_eval_episodes_for_front: int,
    eval_mode="ser",
    scalarization=weighted_sum,
    log=True,
):
    """Evaluates a policy.

    Args:
        policy: to evaluate
        eval_env: environment to evaluate on
        num_eval_episodes_for_front: number of episodes to evaluate on
    Return:
            the discounted returns of the policy
    """
    _, _, reward, discounted_reward = policy.wrapped.policy_eval(
        eval_env,
        num_episodes=num_eval_episodes_for_front,
        weights=np.array([0.25] * 4),
        scalarization=scalarization,
        log=False,
    )
    return reward, discounted_reward


def log_all_multi_policy_metrics(
    current_front: List[np.ndarray],
    hv_ref_point: np.ndarray,
    reward_dim: int,
    n_sample_weights: int,
    ref_front: Optional[List[np.ndarray]] = None,
):
    """Logs all metrics for multi-policy training.

    Logged metrics:
    - hypervolume
    - expected utility metric (EUM)
    If a reference front is provided, also logs:
    - Inverted generational distance (IGD)
    - Maximum utility loss (MUL)

    Args:
        current_front (List) : current Pareto front approximation, computed in an evaluation step
        hv_ref_point: reference point for hypervolume computation
        reward_dim: number of objectives
        global_step: global step for logging
        n_sample_weights: number of weights to sample for EUM and MUL computation
        ref_front: reference front, if known
    """
    filtered_front = list(filter_pareto_dominated(current_front))
    hv = hypervolume(hv_ref_point, filtered_front)
    eum = expected_utility(
        filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights)
    )
    card = cardinality(filtered_front)

    metrics_table = wandb.Table(
        columns=["hypervolume", "eum", "cardinality"],
        data=[[hv, eum, card]],
    )

    wandb.log(
        {
            "eval/all_interps_final": metrics_table,
        },
        commit=False,
    )

    front = wandb.Table(
        columns=[f"objective_{i}" for i in range(1, reward_dim + 1)],
        data=[p.tolist() for p in filtered_front],
    )
    wandb.log({"eval/all_interps_front": front})

    # If PF is known, log the additional metrics
    if ref_front is not None:
        generational_distance = igd(
            known_front=ref_front, current_estimate=filtered_front
        )
        mul = maximum_utility_loss(
            front=filtered_front,
            reference_set=ref_front,
            weights_set=get_reference_directions(
                "energy", reward_dim, n_sample_weights
            ).astype(np.float32),
        )
        wandb.log({"eval/igd": generational_distance, "eval/mul": mul})


def eval_all_policies(
    population,
    eval_env: gym.Env,
    ref_point: np.ndarray,
    num_eval_episodes_for_front: int = 5,
    num_eval_weights_for_eval: int = 50,
    known_front: Optional[List[np.ndarray]] = None,
    log=True,
):
    """Evaluates all policies and store their current performances on the buffer and pareto archive."""
    archive = ParetoArchive()
    reward_dim = eval_env.unwrapped.reward_space.shape[0]
    evals = []
    for i, agent in enumerate(population):
        reward, discounted_reward = eval_policy(agent, eval_env, num_eval_episodes_for_front)
        evals.append((reward, discounted_reward))
        # Storing current results
        archive.add(agent, discounted_reward)

    if log:
        log_all_multi_policy_metrics(
            archive.evaluations,
            ref_point,
            reward_dim,
            n_sample_weights=num_eval_weights_for_eval,
            ref_front=known_front,
        )
    return evals, archive
