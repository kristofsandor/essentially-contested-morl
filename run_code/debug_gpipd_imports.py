import sys
import traceback

print("Starting debug imports...")
print("sys.executable:", sys.executable)

try:
    import gymnasium as gym
    import numpy as np
    import wandb
    print("Imported gymnasium, numpy, wandb")
except Exception as e:
    print("Import error (gym/numpy/wandb):", e)
    traceback.print_exc()

try:
    import env  # registers "my-four-room-v0" as side-effect
    print("Imported local env module")
    try:
        print("Registered env entry:", gym.envs.registry.get("my-four-room-v0"))
    except Exception as e:
        print("Error while accessing gym registry:", e)
        traceback.print_exc()
except Exception as e:
    print("Import error (env):", e)
    traceback.print_exc()

try:
    from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPIPD
    print("Imported GPIPD from morl_baselines")
except Exception as e:
    print("Import error (morl_baselines):", e)
    traceback.print_exc()

print("Debug script finished")
