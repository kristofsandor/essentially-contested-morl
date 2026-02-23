from .fire_rescue_env import FireRescueEnv
import gymnasium as gym

# Register the FireRescueEnv with Gymnasium
gym.register(
    id="FireRescue-v0",
    entry_point="env.fire_rescue_env:FireRescueEnv",
    max_episode_steps=100,  # Default max_steps value
)
