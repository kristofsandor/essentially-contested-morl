import gymnasium as gym
from moral_rl.envs.gym_wrapper import GymWrapper
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

import numpy as np

from moral_rl.envs.delivery import N_VASE
from wrappers.delivery_wrapper import WindowObservationWrapper

TOTAL_TIMESTEPS = 100

delivery_config = {
    "env_id": "delivery-v0",
    "layers": ("#", "P", "F", "C", "S", "V"),
    "width": 16,
    "height": 16,
    "num_actions": 9,
    "reward_dim": 4,
}

emergency_config = {
    "env_id": "emergency-v0",
    "layers": ("#", "P", "C", "H", "G"),
    "width": 8,
    "height": 8,
    "num_actions": 9,
    "reward_dim": 2,
}

# register the environment
# gym.register(
#     id="delivery-v0",
#     entry_point="moral_rl.envs.gym_wrapper:GymWrapper",
#     kwargs={"config": delivery_config},
# )

env = GymWrapper(**delivery_config)

agent = PQL(
    env=env,
    ref_point=np.array([0, 0, 0, -N_VASE]),
    gamma=0.99,
    initial_epsilon=1.0,
    epsilon_decay_steps=TOTAL_TIMESTEPS,
    final_epsilon=0,
    seed=None,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    wandb_entity=None,
    log=False,
)

obs, info = env.reset()

pareto_front = agent.train(
    total_timesteps=TOTAL_TIMESTEPS,
    eval_env=env,
    ref_point=np.array([0, 0, 0, -N_VASE]),
    log_every=50,
    action_eval="hypervolume",
)
