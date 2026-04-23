import pickle

import gymnasium as gym
import numpy as np
from morl_baselines.multi_policy.morld.morld import MORLD

import wandb
from env import MyFourRoom
from utils.config import EXPERIMENT_NAME, MAX_EPISODE_LENGTH, TOTAL_TIMESTEPS
from utils.eval import eval_full_four_room
from utils.visualize_front import visualize_front_general
from wrappers import CombineWrapper

# run with all interpretations as objectives
env = gym.wrappers.TimeLimit(
    gym.make("my-four-room-v0"), max_episode_steps=MAX_EPISODE_LENGTH
)
eval_env = gym.wrappers.TimeLimit(
    gym.make("my-four-room-v0"), max_episode_steps=MAX_EPISODE_LENGTH
)

agent = MORLD(
    env=env,
    scalarization_method="ws",
    evaluation_mode="ser",
    policy_name="MOSACDiscrete",
    experiment_name=EXPERIMENT_NAME,
)

agent.train(
    total_timesteps=TOTAL_TIMESTEPS,
    eval_env=eval_env,
    ref_point=np.array([0, 0, 0, 0]),
)
visualize_front_general(
    agent.archive.evaluations,
    columns=["blue_triangle", "blue_circle", "red_triangle", "red_circle"],
)
agent.close_wandb()

# train on weighted scalarization between triangle and circle
from tqdm.auto import tqdm

all_fronts = {}

weights = np.linspace(0.0, 1.0, 10)
for i, w_triangle in enumerate(tqdm(weights, desc="Weight sweep")):
    w_circle = 1.0 - w_triangle
    weight_vec = np.array([w_triangle, w_circle], dtype=np.float32)

    env = CombineWrapper(
        gym.wrappers.TimeLimit(
            gym.make("my-four-room-v0"), max_episode_steps=MAX_EPISODE_LENGTH
        ),
        weight_vec,
    )
    eval_env = CombineWrapper(
        gym.wrappers.TimeLimit(
            gym.make("my-four-room-v0"), max_episode_steps=MAX_EPISODE_LENGTH
        ),
        weight_vec,
    )

    agent = MORLD(
        env=env,
        scalarization_method="ws",
        evaluation_mode="ser",
        policy_name="MOSACDiscrete",
        experiment_name=EXPERIMENT_NAME + f"_w{w_triangle:.2f}:{w_circle:.2f}",
    )

    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=eval_env,
        ref_point=np.array([0, 0]),
    )

    all_fronts[float(w_triangle)] = np.array(agent.archive.evaluations)
    visualize_front_general(
        agent.archive.evaluations,
        columns=["blue_interp", "red_interp"],
    )

    eval_full_four_room(agent)
    agent.close_wandb()

# Optional: inspect all fronts by weight
# write all fronts to a file for later analysis
with open("four_room_fronts.pkl", "wb") as f:
    pickle.dump(all_fronts, f)
