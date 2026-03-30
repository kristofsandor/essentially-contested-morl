from .fire_rescue import FireRescueEnv
import gymnasium as gym

# Register the FireRescueEnv with Gymnasium
gym.register(
    id="FireRescue-v0",
    entry_point="env.fire_rescue:FireRescueEnv",
)
