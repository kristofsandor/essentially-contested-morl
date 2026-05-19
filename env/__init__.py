from .fire_rescue import FireRescueEnv
from .my_four_room import MyFourRoom
import gymnasium as gym

# Register the FireRescueEnv with Gymnasium
# gym.register(
#     id="FireRescue-v0",
#     entry_point="env.fire_rescue:FireRescueEnv",
# )

# gym.envs.registry.pop("my-four-room-v0", None)  # remove existing registration if it exists

# gym.register(
#     id="my-four-room-v0",
#     entry_point="env.my_four_room:MyFourRoom",
# )

gym.register(
    id="goal-safe-v0",
    entry_point="env.reach_goal:ReachGoalEnv",
)