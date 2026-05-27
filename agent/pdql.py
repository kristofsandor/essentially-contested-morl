"""Pareto Deep Q-Learning."""

import numbers
from typing import Any, Callable, List, Optional

import gymnasium as gym
import numpy as np
from torch import nn, optim
import torch as th
import wandb
from utils.weights import random_weights

from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value


class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, action_dim, rew_dim, net_arch):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim
        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_shape[0] + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions

        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            input = th.cat((features, w), dim=features.dim() - 1)
        else:
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            input = th.cat((obs, w), dim=1)
        q_values = self.net(input)
        return q_values.view(
            -1, self.action_dim, self.rew_dim
        )  # Batch size X Actions X Rewards


class PDQL(MOAgent, MOPolicy):
    """Pareto Deep Q-learning.

    Tabular method relying on pareto pruning.
    Paper: K. Van Moffaert and A. Nowé, “Multi-objective reinforcement learning using sets of pareto dominating policies,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
        self,
        env,
        ref_point: np.ndarray,
        gamma: float = 0.8,
        initial_epsilon: float = 1.0,
        epsilon_decay_steps: int = 100000,
        final_epsilon: float = 0.1,
        seed: Optional[int] = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Pareto Q-Learning",
        wandb_entity: Optional[str] = None,
        # from envelope
        device: Union[th.evice, str] = "auto",
        net_arch: Optional[List[int]] = None,
        num_sample_w: int = 10,
        batch_size: int = 64,
        per: bool = False,
        buffer_size: int = 100000,
        log: bool = True,
        learning_starts: int = 1000,
    ):
        """Initialize the Pareto Q-learning algorithm.

        Args:
            env: The environment.
            ref_point: The reference point for the hypervolume metric.
            gamma: The discount factor.
            initial_epsilon: The initial epsilon value.
            epsilon_decay_steps: The number of steps to decay epsilon.
            final_epsilon: The final epsilon value.
            seed: The random seed.
            project_name: The name of the project used for logging.
            experiment_name: The name of the experiment used for logging.
            wandb_entity: The wandb entity used for logging.
            device: The device to use for training.
            net_arch: The network architecture of the Q-network.
            log: Whether to log or not.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.num_sample_w = num_sample_w
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.per = per

        # Algorithm setup
        self.ref_point = ref_point
        self.num_actions = self.action_dim
        self.env_shape = self.observation_shape
        self.num_states = np.prod(self.observation_shape)

        self.num_objectives = self.reward_dim
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.reward_dim))} for _ in range(self.action_dim)]
            for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros(
            (self.num_states, self.num_actions, self.num_objectives)
        )

        self.q_net = QNet(
            self.observation_shape, self.action_dim, self.reward_dim, net_arc=net_arch
        ).to(self.device)
        self.target_q_net = QNet(
            self.observation_shape, self.action_dim, self.reward_dim, net_arch=net_arch
        ).to(self.device)

        self.q_optim = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

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

        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.reward_dim,
                    max_size=buffer_size,
                action_dtype=np.uint8,
            )

    def get_config(self) -> dict:
        """Get the configuration dictionary.

        Returns:
            Dict: A dictionary of parameters and values.
        """
        return {
            "env_id": self.env.unwrapped.spec.id,
            "ref_point": list(self.ref_point),
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "seed": self.seed,
        }

    def score_pareto_cardinality(self, state: int):
        """Compute the action scores based upon the Pareto cardinality metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = self.get_q_set(state)
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, obs: th.Tensor):
        """Compute the action scores based upon the hypervolume metric.

        Args:
            state (int): The current state.

        Returns:
            ndarray: A score per action.
        """
        q_sets = self.get_q_set(obs)
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

    def get_q_set(self, obs, target = False):
        """Compute the Q-set for a given observation.

        Args:
            obs: current observation (array or tensor)

        Returns:
            A set of non-dominated Q vectors (tuples).
        """
        sampled_w = (
            th.tensor(
                random_weights(
                    self.reward_dim,
                    self.num_sample_w,
                    dist="dirichlet",
                    rng=self.np_random,
                    alpha=1.0,
                )
            )
            .float()
            .to(self.device)
        )
        net = self.target_q_net if target else self.q_net
        obs_per_weight = th.stack([obs] * self.num_sample_w)
        q_set = net(obs_per_weight, sampled_w)  # [K, action_dim, reward_dim]
        non_dominated = get_non_dominated(q_set)
        return non_dominated

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

    def calc_non_dominated(self, state: int):
        """Get the non-dominated vectors in a given state.

        Args:
            state (int): The current state.

        Returns:
            Set: A set of Pareto non-dominated vectors.
        """
        # get_q_set -> 1 x objectives
        # candidates -> union of all unique q_sets in all possible actions
        return self.get_q_set(state)

    @th.no_grad()
    def envelope_target(self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor) -> th.Tensor:
        """Computes the envelope target for the given observation and weight.

        Args:
            obs: current observation.
            w: current weight vector.
            sampled_w: set of sampled weight vectors (>1!).

        Returns: the envelope target.
        """
        # Repeat the weights for each sample
        W = sampled_w.repeat(obs.size(0), 1)
        # Repeat the observations for each sampled weight
        next_obs = obs.repeat_interleave(sampled_w.size(0), 0)
        # Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("br,bwar->bwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=2)
        # Max weights in the envelope
        pref = th.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs, W).view(
            obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim
        )

        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3)),
        ).squeeze(2)
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(1, pref.reshape(-1, 1, 1).expand(max_next_q.size(0), 1, max_next_q.size(2))).squeeze(1)
        return max_next_q

    @th.no_grad()
    def ddqn_target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        # Max action for each state
        q_values = self.q_net(obs, w)
        scalarized_q_values = th.einsum("br,bar->ba", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(
            1,
            max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)),
        )
        q_values_target = q_values_target.reshape(-1, self.reward_dim)
        return q_values_target

    def __learn_from_experience(self):
        """Learn from experience by sampling a batch of experiences and updating the Q-network."""
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    b_inds,
                ) = self.__sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self.__sample_batch_experiences()

                target_q = self.hv_target(b_next_obs)
                target_q = b_rewards + (1 - b_dones) * self.gamma * target_q
                q = self.q_net(b_obs, b_actions)
                loss = F.mse_loss()

    
    def hv_target(self, obs: th.Tensor) -> th.Tensor:
        """Computes the hypervolume target for the given observation.
        Choose an action based on the hypervolume improvement of the next state, and then evaluate the Q-values of that action on the target network to get the target Q-values.

        Args:
            obs: current observation.
        Returns: the hypervolume target.
        """
        # Compute the hypervolume for each action
        hv_per_action = self.score_hypervolume(obs)
        max_actions = th.argmax(hv_per_action, dim=1)
        # MO Q-values evaluated on the target network
        target_q_set = self.get_q_set(obs, target=True)  # [K, action_dim, reward_dim]
        # Index the Q-values for the max actions
        target_q = target_q_set[:, max_actions, :]
        return target_q
                
    def get_target_q_set(self, obs):
        """Compute the target Q-set for a given observation.

        Args:
            obs: current observation (array or tensor)
        Returns:
            A set of non-dominated Q vectors (tuples) from the target network.
        """
        self.get_q_set(obs)
            
            

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_eval: int = 50,
        log_every: Optional[int] = 10000,
        action_eval: Optional[str] = "hypervolume",
    ):
        """Learn the Pareto front.

        Args:
            total_timesteps (int, optional): The number of episodes to train for.
            eval_env (gym.Env): The environment to evaluate the policies on.
            eval_ref_point (ndarray, optional): The reference point for the hypervolume metric during evaluation. If none, use the same ref point as training.
            known_pareto_front (List[ndarray], optional): The optimal Pareto front, if known.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            log_every (int, optional): Log the results every number of timesteps. (Default value = 1000)
            action_eval (str, optional): The action evaluation function name. (Default value = 'hypervolume')

        Returns:
            Set: The final Pareto front.
        """
        if action_eval == "hypervolume":
            score_func = self.score_hypervolume
        elif action_eval == "pareto_cardinality":
            score_func = self.score_pareto_cardinality
        else:
            raise Exception("No other method implemented yet")
        if ref_point is None:
            ref_point = self.ref_point
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "log_every": log_every,
                    "action_eval": action_eval,
                }
            )

        while self.global_step < total_timesteps:
            obs, _ = self.env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                if self.global_step < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.select_action(obs, score_func)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.global_step += 1
                self.replay_buffer.add(obs, action, reward, next_obs, terminated)
                if self.global_step > self.learning_starts:
                    self.__learn_from_experience()

                self.counts[obs, action] += 1
                self.non_dominated[obs][action] = self.calc_non_dominated(next_obs)
                self.avg_reward[obs, action] += (
                    reward - self.avg_reward[obs, action]
                ) / self.counts[obs, action]
                obs = next_obs

                if self.log and self.global_step % log_every == 0:
                    wandb.log({"global_step": self.global_step})
                    pf = self._eval_all_policies(eval_env)
                    log_all_multi_policy_metrics(
                        current_front=pf,
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=known_pareto_front,
                    )

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_local_pcs(state=0)

    def _eval_all_policies(self, env: gym.Env) -> List[np.ndarray]:
        """Evaluate all learned policies by tracking them."""
        pf = []
        for vec in self.get_local_pcs(state=0):
            pf.append(self.track_policy(vec, env))

        return pf

    def track_policy(self, vec, env: gym.Env, tol=1e-3):
        """Track a policy from its return vector.

        Args:
            vec (array_like): The return vector to track.
            env (gym.Env): The environment to track the policy in.
            tol (float, optional): The tolerance for the return vector. (Default value = 1e-3)
        """
        target = np.array(vec)
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_rew = np.zeros(self.num_objectives)
        current_gamma = 1.0

        while not (terminated or truncated):
            state = np.ravel_multi_index(state, self.env_shape)
            closest_dist = np.inf
            closest_action = 0
            found_action = False
            new_target = target

            for action in range(self.num_actions):
                im_rew = self.avg_reward[state, action]
                non_dominated_set = self.non_dominated[state][action]

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
            total_rew += current_gamma * reward
            current_gamma *= self.gamma
            target = new_target

        return total_rew

    def get_local_pcs(self, state: int = 0):
        """Collect the local PCS in a given state.

        Args:
            state (int): The state to get a local PCS for. (Default value = 0)

        Returns:
            Set: A set of Pareto optimal vectors.
        """
        return self.get_q_set(state)
