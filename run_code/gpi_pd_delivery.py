import gymnasium as gym
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD

import numpy as np
import wandb

from moral_rl.envs.delivery import N_VASE

STEPS = 100

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
gym.register(
    id="delivery-v0",
    entry_point="moral_rl.envs.gym_wrapper:GymWrapper",
    kwargs={"config": delivery_config},
)

env = gym.make("delivery-v0")

agent = GPIPD(
    env=env,
    log=True,
)

obs, info = env.reset()

agent.train(
    total_timesteps=STEPS,
    eval_env=env,
    ref_point=np.array([0, 0, 0, -N_VASE]),
    eval_freq=10,
)
