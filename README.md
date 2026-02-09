# Multi-Objective Reinforcement Learning under Normative Uncertainty

A research project exploring multi-objective reinforcement learning in ethical decision-making scenarios, focusing on how different ethical theories interpret values like safety and fairness.

## Overview

This project implements gridworld environments for studying multi-objective RL under normative uncertainty, where the same values (safety, fairness) can be measured using different ethical theories.

## Main Environment: Fire Rescue Moral Gridworld

The `MoralGridWorldEnv` simulates a fire rescue scenario where an agent must rescue humans and dogs of different ages from a spreading fire.

### Features

- **Fire spreading**: Probabilistic fire spread across the gridworld
- **Entity types**: Humans (ages 0-100) and dogs (ages 0-15) with varying vulnerability levels
- **Rescue mechanics**: Agent can move and attempt to rescue entities
- **Multi-objective rewards**: 7-dimensional reward vector capturing different ethical interpretations

### Reward Structure

The environment returns a 7-dimensional reward vector:

1. **Task reward**: Number of entities successfully rescued
2. **Safety (Sentient Utilitarianism)**: Maximizes all lives (humans + dogs), including injured entities
3. **Safety (Classical Utilitarianism)**: Maximizes human lives only, including injured humans
4. **Safety (Hedonistic Utilitarianism)**: Maximizes quality-adjusted life years
5. **Fairness (Equal help)**: No one got more help than others
6. **Fairness (Proportional to need)**: Help proportionate to vulnerability
7. **Fairness (Minimum threshold)**: Everyone got some level of help

## Installation

```bash
pip install -e .
```

## Usage

```python
from env import MoralGridWorldEnv

env = MoralGridWorldEnv(
    render_mode="human",
    size=10,
    num_humans=5,
    num_dogs=3,
    max_steps=100,
    fire_spread_prob=0.3,
    initial_fire_cells=3,
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

See `main.py` for a complete example.

## Project Structure

- `env/moral_gridworld.py`: Fire rescue environment with ethical reward interpretations
- `env/multi_objective_grid_world.py`: General multi-objective gridworld environment
- `main.py`: Example usage script
