import gymnasium as gym
from moral_rl.envs.gym_wrapper import GymWrapper
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL

import numpy as np

from moral_rl.envs.delivery import N_VASE

STEPS = 100

delivery_config = {
    'env_id': 'delivery-v0',
    'layers': ('#', 'P', 'F', 'C', 'S', 'V'),
    'width': 16,
    'height': 16,
    'num_actions': 9,
    'reward_dim': 4
}

emergency_config = {
    'env_id': 'emergency-v0',
    'layers': ('#', 'P', 'C', 'H', 'G'),
    'width': 8,
    'height': 8,
    'num_actions': 9,
    'reward_dim': 2
}

# register the environment
gym.register(
    id="delivery-v0",
    entry_point="moral_rl.envs.gym_wrapper:GymWrapper",
    kwargs={"config": delivery_config}
)

env = gym.make("delivery-v0")

agent = PQL(
    env=env,
    ref_point=np.array([0, 0, 0, -N_VASE]),
    gamma=0.8,
    initial_epsilon=1.0,
    epsilon_decay_steps=100000,
    final_epsilon=0.1,
    seed=None,
    project_name="MORL-Baselines",
    experiment_name="Pareto Q-Learning",
    wandb_entity=None,
    log=True,
)

obs, info = env.reset()

for i in range(STEPS):
    action = agent.act(obs)
    obs, vector_reward, terminated, truncated, info = env.step(action)
