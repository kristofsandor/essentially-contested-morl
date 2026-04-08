from .fire_rescue import FireRescueEnv
from .my_four_room import MyFourRoom
import gymnasium as gym

# Register the FireRescueEnv with Gymnasium
gym.register(
    id="FireRescue-v0",
    entry_point="env.fire_rescue:FireRescueEnv",
)

gym.register(
    id="my-four-room-v0",
    entry_point="env.my_four_room:MyFourRoom",
)