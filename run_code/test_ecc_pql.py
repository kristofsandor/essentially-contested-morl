"""Quick test of ECC-PQL to verify it works."""

import numpy as np
from env import FireRescueEnv
from agent.ecc_pql import ECCPQL


def test():
    print("Creating environment...")
    env = FireRescueEnv(
        render_mode=None,
        size=5,  # Smaller for faster testing
        num_humans=2,
        num_dogs=1,
        max_steps=20,
        fire_spread_prob=0.2,
        initial_fire_cells=1,
    )

    print("Creating agent...")
    safety_ref_point = np.array([0.0, 0.0, 0.0])
    fairness_ref_point = np.array([0.0, 0.0, 0.0])

    agent = ECCPQL(
        env=env,
        safety_ref_point=safety_ref_point,
        fairness_ref_point=fairness_ref_point,
        gamma=0.9,
        initial_epsilon=1.0,
        epsilon_decay_steps=100,
        final_epsilon=0.1,
        seed=42,
        log=False,
    )

    print("Starting training (1000 timesteps)...")
    value_pareto_front = agent.train(
        total_timesteps=1000,
        eval_env=env,
        log_every=500,
        action_eval="hypervolume",
    )

    print(f"\nTraining completed!")
    print(f"Value Pareto front: {len(value_pareto_front)} policies")
    for i, pair in enumerate(value_pareto_front):
        print(f"  Policy {i+1}: Safety_hv={pair[0]:.4f}, Fairness_hv={pair[1]:.4f}")

    env.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    test()
