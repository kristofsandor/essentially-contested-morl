# eval.py

from email import policy
from typing import Optional, List
import gymnasium as gym
from morl_baselines.common.evaluation import (
    cardinality,
    expected_utility,
    hypervolume,
    igd,
    maximum_utility_loss,
)
from morl_baselines.common.pareto import ParetoArchive, filter_pareto_dominated, get_non_dominated_inds
from morl_baselines.common.scalarization import weighted_sum
from morl_baselines.common.weights import equally_spaced_weights
import numpy as np

from morl_baselines.multi_policy.morld.morld import Policy
from pymoo.util.ref_dirs import get_reference_directions
import wandb

from utils.config import MAX_EPISODE_LENGTH
from utils.visualize_front import visualize_front_general

def eval_full_room_gpipd(agent, n_weights=25, n_episodes_per_weight=5, filename="", render=False):
    """Evaluates for different weights"""
    eval_weights = equally_spaced_weights(agent.reward_dim, n=n_weights)
    test_env = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0", render_mode="human" if render else None), max_episode_steps=MAX_EPISODE_LENGTH, 
    )
    # get weight of the policy and the corresponding returns for each policy in the population
    returns = np.zeros((len(eval_weights), 10)) #  2 for weight, 4 for avg_vec_return, 4 for avg_disc_vec_return

    for j, weight in enumerate(eval_weights):
        # evaluate the policy: get averages over n_evaluation_episodes episodes
        policy_evals = np.zeros((n_episodes_per_weight, 8))  # to store (vec_return, disc_vec_return) for each episode
        for i in range(n_episodes_per_weight):
            obs, _ = test_env.reset()
            done = False
            vec_return, disc_vec_return = np.zeros(4), np.zeros(4)
            gamma = 1.0

            while not done:
                obs, r, terminated, truncated, info = test_env.step(agent.eval(obs, weight))
                done = terminated or truncated
                vec_return += r
                disc_vec_return += gamma * r
                gamma *= agent.gamma

            policy_evals[i, 0:4] = vec_return
            policy_evals[i, 4:8] = disc_vec_return

        avg_vec_return = np.mean(policy_evals[:, 0:4], axis=0)
        avg_disc_vec_return = np.mean(policy_evals[:, 4:8], axis=0)
        returns[j, 0:2] = weight
        returns[j, 2:6] = avg_vec_return
        returns[j, 6:10] = avg_disc_vec_return

    # log returns for each policy in the population as a wandb table
    returns_table = wandb.Table(columns=["weight_blue", "weight_red", *[f"avg_vec_return_{i}" for i in range(4)], *[f"avg_disc_vec_return_{i}" for i in range(4)]], data=returns)
    wandb.log({"policy_returns": returns_table})

    pf_idxs = get_non_dominated_inds(returns[:, 2:6])  # get non-dominated indices based on avg_vec_return
    returns_pf = returns[pf_idxs][:, 2:6]  # get the avg_vec_return of the non-dominated policies

    visualize_front_general(returns_pf, columns=["blue_triangle", "blue_circle", "red_triangle", "red_circle"], filename=filename)

    log_all_multi_policy_metrics(
        current_front=returns_pf,
        hv_ref_point=np.array([0, 0, 0, 0]),
        reward_dim=4,
        n_sample_weights=50,
    )


def eval_full_four_room(agent, n_evaluation_episodes=5):
    test_env = gym.wrappers.TimeLimit(
        gym.make("my-four-room-v0", render_mode="human"), max_episode_steps=MAX_EPISODE_LENGTH, 
    )
    # get weight of the policy and the corresponding returns for each policy in the population
    returns = np.zeros((len(agent.population), 10), dtype=object) #  2 for weight, 4 for avg_vec_return, 4 for avg_disc_vec_return

    for j, policy in enumerate(agent.population):
        # evaluate the policy: get averages over n_evaluation_episodes episodes
        policy_evals = np.zeros((n_evaluation_episodes, 8))  # to store (vec_return, disc_vec_return) for each episode
        for i in range(n_evaluation_episodes):
            obs, _ = test_env.reset()
            done = False
            vec_return, disc_vec_return = np.zeros(4), np.zeros(4)
            gamma = 1.0

            while not done:
                obs, r, terminated, truncated, info = test_env.step(policy.wrapped.eval(obs))
                done = terminated or truncated
                vec_return += r
                disc_vec_return += gamma * r
                gamma *= policy.wrapped.gamma

            policy_evals[i, 0:4] = vec_return
            policy_evals[i, 4:8] = disc_vec_return

        avg_vec_return = np.mean(policy_evals[:, 0:4], axis=0)
        avg_disc_vec_return = np.mean(policy_evals[:, 4:8], axis=0)
        returns[j, 0:2] = policy.weights
        returns[j, 2:6] = avg_vec_return
        returns[j, 6:10] = avg_disc_vec_return

    # log returns for each policy in the population as a wandb table
    returns_table = wandb.Table(columns=["weight_blue", "weight_red", *[f"avg_vec_return_{i}" for i in range(4)], *[f"avg_disc_vec_return_{i}" for i in range(4)]], data=returns)
    wandb.log({"policy_returns": returns_table})

    pf_idxs = get_non_dominated_inds(returns[:, 2:6])  # get non-dominated indices based on avg_vec_return
    returns_pf = returns[pf_idxs][:, 2:6]  # get the avg_vec_return of the non-dominated policies

    visualize_front_general(returns_pf, columns=["blue_triangle", "blue_circle", "red_triangle", "red_circle"])

    log_all_multi_policy_metrics(
        current_front=returns_pf,
        hv_ref_point=np.array([0, 0, 0, 0]),
        reward_dim=4,
        n_sample_weights=50,
    )


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
