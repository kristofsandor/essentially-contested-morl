"""Quick test of PQL to verify it works with MO-Gymnasium environments."""

import numpy as np
import mo_gymnasium as mo_gym

from morl_baselines.multi_policy.pareto_q_learning.pql import PQL


def test():
    print("Creating MO-Gymnasium environment...")
    # Create DeepSeaTreasure environment from MO-Gymnasium
    env = mo_gym.make("deep-sea-treasure-v0")
    
    print(f"Environment: {env.unwrapped.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Reward space: {env.unwrapped.reward_space}")
    print(f"Number of objectives: {env.unwrapped.reward_space.shape[0]}")

    print("\nCreating PQL agent...")
    # Reference point for hypervolume calculation (should be worse than worst possible return)
    # For DeepSeaTreasure: [time_penalty (negative), treasure_value (positive)]
    # Using a point worse than worst case: very negative time penalty, zero treasure
    ref_point = np.array([-100.0, 0.0])

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=0.99,
        initial_epsilon=1.0,
        epsilon_decay_steps=100,
        final_epsilon=0.1,
        seed=42,
        log=False,
    )

    print("Starting training (50 timesteps)...")
    pareto_front = agent.train(
        total_timesteps=50,
        eval_env=env,
        ref_point=ref_point,
        log_every=1000,
        action_eval="hypervolume",
    )

    print(f"\nTraining completed!")
    print(f"Pareto front: {len(pareto_front)} policies")
    for i, vec in enumerate(pareto_front):
        print(f"  Policy {i+1}: {vec}")

    env.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
