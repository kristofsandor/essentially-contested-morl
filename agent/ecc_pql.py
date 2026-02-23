"""Essentially Contested Concept Pareto Q-Learning (ECC-PQL)."""

import numbers
import time
from typing import Callable, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value

import wandb


class ECCPQL(MOAgent):
    """Essentially Contested Concept Pareto Q-learning.

    This algorithm learns Pareto fronts within Safety and Fairness separately,
    then aggregates them using hypervolume to find a Pareto front over (Safety, Fairness) value pairs.

    Safety objectives (indices 1-3): sentient, classical, hedonistic utilitarianism
    Fairness objectives (indices 4-6): equal, proportional, minimum threshold

    Paper: Based on K. Van Moffaert and A. Nowé, "Multi-objective reinforcement learning
    using sets of pareto dominating policies," The Journal of Machine Learning Research,
    vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        safety_ref_point: np.ndarray,
        fairness_ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        project_name: str = "ECC-MORL",
        experiment_name: str = "ECC-Pareto Q-Learning",
        wandb_entity: Optional[str] = None,
        log: bool = True,
    ):
        """Initialize the ECC-PQL algorithm.

        Args:
            env: The environment.
            safety_ref_point: The reference point for Safety hypervolume (3D).
            fairness_ref_point: The reference point for Fairness hypervolume (3D).
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            log: Whether to log or not.
        """
        # Initialize MOAgent - we'll override extract_env_info to handle Dict spaces
        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup - contested concepts
        self.safety_ref_point = np.array(safety_ref_point)
        self.fairness_ref_point = np.array(fairness_ref_point)

        # Safety objectives: indices [1, 2, 3] (sentient, classical, hedonistic)
        # Fairness objectives: indices [4, 5, 6] (equal, proportional, minimum)
        self.safety_objectives = [1, 2, 3]
        self.fairness_objectives = [4, 5, 6]
        self.num_safety_objectives = 3
        self.num_fairness_objectives = 3

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("ECC-PQL only supports (multi)discrete action spaces.")

        # Handle different observation space types
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
            self._state_to_int = lambda obs: int(obs)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
            self._state_to_int = lambda obs: int(
                np.ravel_multi_index(obs, self.env_shape)
            )
        elif isinstance(self.env.observation_space, gym.spaces.Dict):
            # For Dict spaces, use agent position as state representation
            # This assumes the Dict has an "agent" key with position information
            if "agent" in self.env.observation_space.spaces:
                agent_space = self.env.observation_space.spaces["agent"]
                if isinstance(agent_space, gym.spaces.Box) and agent_space.shape == (
                    2,
                ):
                    # Agent position is 2D (row, col)
                    # Extract size from environment if available, otherwise estimate
                    if hasattr(self.env, "size"):
                        size = self.env.size
                    else:
                        # Try to infer from observation space bounds
                        size = int(agent_space.high[0]) + 1
                    self.env_shape = (size, size)
                    self._state_to_int = lambda obs: int(
                        np.ravel_multi_index(obs["agent"], self.env_shape)
                    )
                else:
                    raise Exception(
                        "ECC-PQL Dict observation space must have 'agent' key with Box(2,) shape."
                    )
            else:
                raise Exception("ECC-PQL Dict observation space must have 'agent' key.")
        elif (
            isinstance(self.env.observation_space, gym.spaces.Box)
            and self.env.observation_space.is_bounded(manner="both")
            and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
            self._state_to_int = lambda obs: int(
                np.ravel_multi_index(obs, self.env_shape)
            )
        else:
            raise Exception("ECC-PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)

        self.num_objectives = self.env.unwrapped.reward_space.shape[0]
        # Get number of objectives from reward_space if available, otherwise assume 7

        # Counts for averaging
        self.counts = np.zeros((self.num_states, self.num_actions))

        # Safety contested concept tracking over states, actions
        self.safety_non_dominated = [
            [
                {tuple(np.zeros(self.num_safety_objectives))}
                for _ in range(self.num_actions)
            ]
            for _ in range(self.num_states)
        ]
        self.safety_avg_reward = np.zeros(
            (self.num_states, self.num_actions, self.num_safety_objectives)
        )

        # Fairness contested concept tracking over states, actions
        self.fairness_non_dominated = [
            [
                {tuple(np.zeros(self.num_fairness_objectives))}
                for _ in range(self.num_actions)
            ]
            for _ in range(self.num_states)
        ]
        self.fairness_avg_reward = np.zeros(
            (self.num_states, self.num_actions, self.num_fairness_objectives)
        )

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log

        if self.log:
            self.setup_wandb(
                project_name=self.project_name,
                experiment_name=self.experiment_name,
                entity=wandb_entity,
            )

    # def setup_wandb(
    #     self,
    #     project_name: str,
    #     experiment_name: str,
    #     entity: Optional[str] = None,
    #     group: Optional[str] = None,
    #     mode: str = "online",
    # ) -> None:
    #     """Initialize wandb writer, handling environments without spec.

    #     Args:
    #         project_name: name of the wandb project.
    #         experiment_name: name of the wandb experiment.
    #         entity: wandb entity.
    #         group: optional group name.
    #         mode: wandb mode.
    #     """
    #     self.experiment_name = experiment_name

    #     # Handle environments without spec (like FireRescueEnv)
    #     if self.env is None or not hasattr(self.env, 'spec') or self.env.spec is None:
    #         env_id = "FireRescueEnv"
    #     elif isinstance(self.env, gym.vector.SyncVectorEnv):
    #         env_id = self.env.envs[0].spec.id if hasattr(self.env.envs[0], 'spec') and self.env.envs[0].spec else "UnknownEnv"
    #     else:
    #         env_id = self.env.spec.id if hasattr(self.env.spec, 'id') else "UnknownEnv"

    #     self.full_experiment_name = f"{env_id}__{experiment_name}__{self.seed}__{int(time.time())}"
    #     import wandb

    #     config = self.get_config()
    #     config["algo"] = self.experiment_name
    #     # looks for whether we're using a Gymnasium based env in env_variable
    #     import os
    #     from distutils.util import strtobool
    #     monitor_gym = strtobool(os.environ.get("MONITOR_GYM", "True"))

    #     wandb.init(
    #         project=project_name,
    #         entity=entity,
    #         config=config,
    #         name=self.full_experiment_name,
    #         monitor_gym=monitor_gym,
    #         save_code=True,
    #         group=group,
    #         mode=mode,
    #         settings=wandb.Settings(init_timeout=120, _disable_stats=True),
    #     )
    #     # The default "step" of wandb is not the actual time step (global_step) of the MDP
    #     wandb.define_metric("*", step_metric="global_step")

    def extract_env_info(self, env: Optional[gym.Env]) -> None:
        """Extracts all the features of the environment, handling Dict observation spaces.

        Args:
            env (gym.Env): The environment
        """
        if env is not None:
            self.env = env
            # Handle Dict observation spaces (used by FireRescueEnv)
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                # For Dict spaces, set dummy values since we handle them separately
                self.observation_shape = None
                self.observation_dim = None
            elif isinstance(self.env.observation_space, gym.spaces.Discrete):
                self.observation_shape = (1,)
                self.observation_dim = self.env.observation_space.n
            else:
                self.observation_shape = self.env.observation_space.shape
                self.observation_dim = self.env.observation_space.shape[0]

            self.action_space = env.action_space
            if isinstance(
                self.env.action_space, (gym.spaces.Discrete, gym.spaces.MultiBinary)
            ):
                self.action_shape = (1,)
                self.action_dim = self.env.action_space.n
            else:
                self.action_shape = self.env.action_space.shape
                self.action_dim = self.env.action_space.shape[0]

            # Handle reward_dim - use reward_space if available, otherwise assume 7
            if hasattr(self.env.unwrapped, "reward_space"):
                self.reward_dim = self.env.unwrapped.reward_space.shape[0]
            else:
                self.reward_dim = 7  # Default for FireRescueEnv

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        # Handle environments without spec
        if self.env is None:
            env_id = "UnknownEnv"
        elif (
            hasattr(self.env, "unwrapped")
            and hasattr(self.env.unwrapped, "spec")
            and self.env.unwrapped.spec is not None
        ):
            env_id = self.env.unwrapped.spec.id
        elif hasattr(self.env, "spec") and self.env.spec is not None:
            env_id = self.env.spec.id
        else:
            env_id = "FireRescueEnv"  # Default for our custom environment

        return {
            "env_id": env_id,
            "safety_ref_point": list(self.safety_ref_point),
            "fairness_ref_point": list(self.fairness_ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def get_q_set(
        self, avg_reward, non_dominated, state: int, action: int
    ) -> Set[Tuple]:
        """Compute the Q-set for a given state-action pair.

        Args:
            avg_reward (np.ndarray): The average reward.
            non_dominated (Set[Tuple]): The non-dominated vectors.
            state (int): The current state.
            action (int): The action.
        """
        discounted_non_dominated = self.gamma * np.array(
            list(non_dominated[state][action])
        )
        q_array = avg_reward[state, action] + discounted_non_dominated
        return {tuple(vec) for vec in q_array}

    def calc_non_dominated(self, avg_reward, non_dominated, state: int) -> Set[Tuple]:
        """Calculate the non-dominated vectors in a given state.

        Args:
            avg_reward (np.ndarray): The average reward.
            non_dominated (Set[Tuple]): The non-dominated vectors.
            state (int): The current state.

        Returns:
            Set[Tuple]: The non-dominated vectors.
        """

        candidates = set().union(*[self.get_q_set(avg_reward, non_dominated, state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def score_hypervolume(self, state: int) -> np.ndarray:
        """Compute the action scores based upon combined Safety and Fairness hypervolumes.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action (product of Safety and Fairness hypervolumes).
        """
        action_scores = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            safety_q_set = self.get_q_set(self.safety_avg_reward, self.safety_non_dominated, state, action)
            fairness_q_set = self.get_q_set(self.fairness_avg_reward, self.fairness_non_dominated, state, action)

            safety_hv = hypervolume(self.safety_ref_point, list(safety_q_set))
            fairness_hv = hypervolume(self.fairness_ref_point, list(fairness_q_set))

            # Use product to encourage balanced exploration of both contested concepts
            action_scores[action] = safety_hv * fairness_hv
        return action_scores

    def score_pareto_cardinality(self, state: int) -> np.ndarray:
        """Compute the action scores based upon Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        safety_q_sets = np.array(
            [
                self.get_q_set(
                    self.safety_avg_reward, self.safety_non_dominated, state, action
                )
                for action in range(self.num_actions)
            ]
        )
        fairness_q_sets = np.array(
            [
                self.get_q_set(
                    self.fairness_avg_reward, self.fairness_non_dominated, state, action
                )
                for action in range(self.num_actions)
            ]
        )

        safety_non_dominated = get_non_dominated(safety_q_sets)
        fairness_non_dominated = get_non_dominated(fairness_q_sets)

        scores = np.zeros(self.num_actions)

        for vec in safety_non_dominated:
            for action, q_set in enumerate(safety_q_sets):
                if vec in q_set:
                    scores[action] += 1

        for vec in fairness_non_dominated:
            for action, q_set in enumerate(fairness_q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def select_action(self, state: int, score_func: Callable):
        """Select an action in the current state.

        Args:
            state (int): The current state.
            score_func (callable): A function that returns a score per action.

        Returns:
            int: The selected action.
        """
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(
                np.argwhere(action_scores == np.max(action_scores)).flatten()
            )

    def get_local_safety_pcs(self, state: int = 0) -> Set[Tuple]:
        """Collect the local Safety Pareto Coverage Set in a given state.

        Args:
            state (int): The state to get a local Safety PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal Safety vectors (3D).
        """
        q_sets = [
            self.get_q_set(
                self.safety_avg_reward, self.safety_non_dominated, state, action
            )
            for action in range(self.num_actions)
        ]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)

    def get_local_fairness_pcs(self, state: int = 0) -> Set[Tuple]:
        """Collect the local Fairness Pareto Coverage Set in a given state.

        Args:
            state (int): The state to get a local Fairness PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal Fairness vectors (3D).
        """
        q_sets = [
            self.get_q_set(
                self.fairness_avg_reward, self.fairness_non_dominated, state, action
            )
            for action in range(self.num_actions)
        ]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)

    def get_value_pareto_front(self, state: int = 0) -> List[np.ndarray]:
        """Get Pareto front over (Safety_hypervolume, Fairness_hypervolume) pairs.

        For each policy in the Safety and Fairness Pareto fronts, compute its
        hypervolume values and find Pareto optimal (Safety_hv, Fairness_hv) pairs.

        Args:
            state (int): The state to get value Pareto front for. (Default value = 0)

        Returns:
            List[np.ndarray]: List of (Safety_hv, Fairness_hv) pairs on the Pareto front.
        """
        # Get all possible policies from Safety and Fairness Pareto fronts
        # safety_pcs = self.get_local_safety_pcs(state)
        # fairness_pcs = self.get_local_fairness_pcs(state)

        # For each policy, we need to track it and compute both Safety and Fairness hypervolumes
        # Since we're tracking policies implicitly through Q-sets, we'll evaluate
        # all combinations of Safety and Fairness policies
        value_pairs = []

        # Collect all state-action pairs and their value pairs
        for action in range(self.num_actions):
            safety_q_set = self.get_q_set(
                self.safety_avg_reward, self.safety_non_dominated, state, action
            )
            fairness_q_set = self.get_q_set(
                self.fairness_avg_reward, self.fairness_non_dominated, state, action
            )

            # Compute hypervolumes for this action
            if len(safety_q_set) > 0:
                safety_hv = hypervolume(self.safety_ref_point, list(safety_q_set))
            else:
                safety_hv = 0.0

            if len(fairness_q_set) > 0:
                fairness_hv = hypervolume(self.fairness_ref_point, list(fairness_q_set))
            else:
                fairness_hv = 0.0

            value_pairs.append(np.array([safety_hv, fairness_hv]))

        # Find Pareto optimal value pairs
        if len(value_pairs) == 0:
            return []

        value_pairs_set = {tuple(pair) for pair in value_pairs}
        pareto_optimal = get_non_dominated(value_pairs_set)

        return [np.array(pair) for pair in pareto_optimal]

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        safety_ref_point: Optional[np.ndarray] = None,
        fairness_ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        log_every: Optional[int] = 10000,
        action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto fronts for contested concepts.

        Args:
            total_timesteps (int): The number of timesteps to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            safety_ref_point (ndarray, optional): The reference point for Safety hypervolume during evaluation.
            fairness_ref_point (ndarray, optional): The reference point for Fairness hypervolume during evaluation.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 10000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            List[np.ndarray]: The final value-level Pareto front (Safety_hv, Fairness_hv) pairs.
        """
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")

        if safety_ref_point is None:
            safety_ref_point = self.safety_ref_point
        if fairness_ref_point is None:
            fairness_ref_point = self.fairness_ref_point

        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "safety_ref_point": safety_ref_point.tolist(),
                    "fairness_ref_point": fairness_ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "log_every": log_every,
                    "action_eval": action_eval,
                }
            )

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            state = self._state_to_int(state)
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                next_state = self._state_to_int(next_state)

                # Extract Safety and Fairness rewards
                safety_reward = reward[self.safety_objectives]
                fairness_reward = reward[self.fairness_objectives]

                self.counts[state, action] += 1

                # Update Safety contested concept
                self.safety_non_dominated[state][action] = self.calc_non_dominated(
                    self.safety_avg_reward, self.safety_non_dominated, next_state
                )
                self.safety_avg_reward[state, action] += (
                    safety_reward - self.safety_avg_reward[state, action]
                ) / self.counts[state, action]

                # Update Fairness contested concept
                self.fairness_non_dominated[state][action] = self.calc_non_dominated(
                    self.fairness_avg_reward, self.fairness_non_dominated, next_state
                )
                self.fairness_avg_reward[state, action] += (
                    fairness_reward - self.fairness_avg_reward[state, action]
                ) / self.counts[state, action]

                state = next_state

                if self.log and self.global_step % log_every == 0:
                    wandb.log({"global_step": self.global_step})

                    # Evaluate all policies
                    safety_pf, fairness_pf, value_pf = self._eval_all_policies(eval_env)

                    # Log Safety Pareto front (3D)
                    if len(safety_pf) > 0:
                        safety_pf_array = np.array(safety_pf)
                        for i in range(self.num_safety_objectives):
                            wandb.log(
                                {f"safety_pf/dim_{i}": safety_pf_array[:, i].tolist()}
                            )
                        wandb.log({"safety_pf/cardinality": len(safety_pf)})

                    # Log Fairness Pareto front (3D)
                    if len(fairness_pf) > 0:
                        fairness_pf_array = np.array(fairness_pf)
                        for i in range(self.num_fairness_objectives):
                            wandb.log(
                                {
                                    f"fairness_pf/dim_{i}": fairness_pf_array[
                                        :, i
                                    ].tolist()
                                }
                            )
                        wandb.log({"fairness_pf/cardinality": len(fairness_pf)})

                    # Log Value-level Pareto front (2D: Safety_hv vs Fairness_hv)
                    if len(value_pf) > 0:
                        value_pf_array = np.array(value_pf)
                        wandb.log(
                            {
                                "value_pf/safety_hypervolume": value_pf_array[
                                    :, 0
                                ].tolist(),
                                "value_pf/fairness_hypervolume": value_pf_array[
                                    :, 1
                                ].tolist(),
                                "value_pf/cardinality": len(value_pf),
                            }
                        )

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_value_pareto_front(state=0)

    def _eval_all_policies(
        self, env: gym.Env
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Evaluate all learned policies by tracking them.

        Returns:
            Tuple of (Safety Pareto front, Fairness Pareto front, Value Pareto front).
        """
        safety_pf = []
        fairness_pf = []
        value_pairs = []

        # Get Safety and Fairness Pareto fronts
        safety_pcs = self.get_local_safety_pcs(state=0)
        fairness_pcs = self.get_local_fairness_pcs(state=0)

        # Track policies for Safety
        for safety_vec in safety_pcs:
            safety_result = self.track_policy_safety(np.array(safety_vec), env)
            safety_pf.append(safety_result)

        # Track policies for Fairness
        for fairness_vec in fairness_pcs:
            fairness_result = self.track_policy_fairness(np.array(fairness_vec), env)
            fairness_pf.append(fairness_result)

        # Compute value-level Pareto front
        value_pf = self.get_value_pareto_front(state=0)

        return safety_pf, fairness_pf, value_pf

    def track_policy_safety(
        self, vec: np.ndarray, env: gym.Env, tol: float = 1e-3
    ) -> np.ndarray:
        """Track a Safety policy from its return vector.

        Args:
            vec (array_like): The Safety return vector to track (3D).
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)

        Returns:
            np.ndarray: Total Safety reward (3D).
        """
        target = np.array(vec)
        state, _ = env.reset()
        state = self._state_to_int(state)
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_safety_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.safety_avg_reward[state, action]
                non_dominated_set = self.safety_non_dominated[state, action]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            state = self._state_to_int(state)
            safety_reward = reward[self.safety_objectives]
            total_rew += current_gamma * safety_reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def track_policy_fairness(
        self, vec: np.ndarray, env: gym.Env, tol: float = 1e-3
    ) -> np.ndarray:
        """Track a Fairness policy from its return vector.

        Args:
            vec (array_like): The Fairness return vector to track (3D).
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)

        Returns:
            np.ndarray: Total Fairness reward (3D).
        """
        target = np.array(vec)
        state, _ = env.reset()
        state = self._state_to_int(state)
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_fairness_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.fairness_avg_reward[state, action]
                non_dominated_set = self.fairness_non_dominated[state][action]

                for q in non_dominated_set:
                    q = np.array(q)
                    dist = np.sum(np.abs(self.gamma * q + im_rew - target))
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_action = action
                        new_target = q

                        if dist < tol:
                            found_action = True
                            break

                if found_action:
                    break

            state, reward, terminated, truncated, _ = env.step(closest_action)
            state = self._state_to_int(state)
            fairness_reward = reward[self.fairness_objectives]
            total_rew += current_gamma * fairness_reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew
