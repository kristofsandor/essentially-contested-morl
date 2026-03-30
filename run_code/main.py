"""
This demonstrates how to use the fire rescue environment and interpret the multi-objective
rewards according to different ethical theories.
"""

import numpy as np
import pygame

# import wandb
from env import FireRescueEnv

# wandb.init(project="ecc-morl")


def wait_for_space():
    """Wait for space bar press, return True when pressed. Returns False if window is closed."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Window closed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True
        pygame.time.wait(10)  # Small delay to avoid busy-waiting


def main():
    # Create the environment
    env = FireRescueEnv(
        render_mode="human",
        size=5,
        num_humans=2,
        num_dogs=1,
        max_steps=1000,
        fire_spread_prob=0.3,
        initial_fire_cells=1,
    )

    # Reset the environment
    obs, info = env.reset()

    print("Environment reset!")
    print(f"Observation keys: {obs.keys()}")
    print(f"Info keys: {info.keys()}")
    print("\nReward structure:")
    print("[task_reward, safety_sentient, safety_classical, safety_hedonistic,")
    print(" fairness_equal, fairness_proportional, fairness_minimum]")
    print("\nStarting episode...\n")

    total_rewards = np.zeros(7)
    step = 0
    max_steps = 50
    running = True

    print("Press SPACE to advance each step. Close window to exit.\n")

    # Run steps manually - wait for space bar before each step
    while running and step < max_steps:
        # Wait for space bar before stepping
        if not wait_for_space():
            running = False
            break

        # Random action
        action = env.action_space.sample()
        obs, reward_vector, terminated, truncated, info = env.step(action)

        total_rewards += reward_vector
        step += 1

        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Task reward (rescued): {total_rewards[0]:.2f}")
            print(f"  Safety (Sentient Utilitarianism): {total_rewards[1]:.2f}")
            print(f"  Safety (Classical Utilitarianism): {total_rewards[2]:.2f}")
            print(f"  Safety (Hedonistic Utilitarianism): {total_rewards[3]:.2f}")
            print(f"  Fairness (Equal help): {total_rewards[4]:.2f}")
            print(f"  Fairness (Proportional to need): {total_rewards[5]:.2f}")
            print(f"  Fairness (Minimum threshold): {total_rewards[6]:.2f}")
            print(
                f"  Rescued: {info['rescued_count']} | Injured: {info['injured_count']} | Dead: {info['dead_count']}"
            )
            print(f"  Fire cells: {info['fire_cells']}")

        if terminated or truncated:
            print("\nEpisode finished!")
            break

    print("\nFinal episode statistics:")
    print(f"Total rewards: {total_rewards}")
    print(f"Final info: {info}")

    env.close()


if __name__ == "__main__":
    main()
