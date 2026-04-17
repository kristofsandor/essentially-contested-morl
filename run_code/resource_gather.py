import gymnasium as gym
from morl_baselines.multi_policy.multi_policy_moqlearning.mp_mo_q_learning import (
    MPMOQLearning,
)
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers.vector.wrappers import MOSyncVectorEnv, MORecordEpisodeStatistics
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import numpy as np


GAMMA = 0.9

ref_point = np.array([-1, -1, -2])

# env = mo_gym.make_vec("resource-gathering-v0", num_envs=1, vectorization_mode="sync", wrappers=(MORecordEpisodeStatistics,))
# env = MOSyncVectorEnv([lambda: mo_gym.make("resource-gathering-v0")])
# env = MORecordEpisodeStatistics(env, gamma=GAMMA)
env = mo_gym.make("resource-gathering-v0")
from mo_gymnasium.envs.resource_gathering import ResourceGatheringEnv

eval_env = mo_gym.make("resource-gathering-v0")

obs, info = env.reset()


# Your code here:
agent = MPMOQLearning(
    env,
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=100000,
    gamma=GAMMA,
    dyna=True,
    gpi_pd=True,
    weight_selection_algo="gpi-ls",
    use_gpi_policy=True,
    log=False,
)

agent.train(
    total_timesteps=100000,
    timesteps_per_iteration=10000,
    eval_env=eval_env,
    num_eval_episodes_for_front=50,
    ref_point=ref_point,
)
