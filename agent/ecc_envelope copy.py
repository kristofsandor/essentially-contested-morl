"""Envelope Q-Learning implementation."""

import os
from pathlib import Path
import time
from collections import deque
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import hypervolume, log_all_multi_policy_metrics
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import (
    NatureCNN,
    get_grad_norm,
    layer_init,
    mlp,
    polyak_update,
)
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.prioritized_buffer import PrioritizedReplayBuffer
from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.weights import equally_spaced_weights
from typing_extensions import override

import wandb
from networks.qnet import CNNQNet


def random_weights(
    dim: int,
    n: int = 1,
    dist: str = "dirichlet",
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    alpha=1.0,
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian' or 'dirichlet'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim) * alpha, n)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w


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


class QNetEnsemble(nn.ModuleList):
    """A list of Q-networks callable to infer over all of them at once."""

    def forward(self, obs, w):
        """Run every Q-net and stack the results.

        Returns: a tensor of shape [num_interps, ...] with the stacked Q-values.
        """
        return th.stack([q_net(obs, w) for q_net in self])


class ECCEnvelope(MOPolicy, MOAgent):
    """Envelope Q-Leaning Algorithm.

    Envelope uses a conditioned network to embed multiple policies (taking the weight as input).
    The main change of this algorithm compare to a scalarized CN DQN is the target update.
    Paper: R. Yang, X. Sun, and K. Narasimhan, “A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation,” arXiv:1908.08342 [cs], Nov. 2019, Accessed: Sep. 06, 2021. [Online]. Available: http://arxiv.org/abs/1908.08342.
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 0.01,
        final_epsilon: float = 0.01,
        epsilon_decay_steps: int = None,  # None == fixed epsilon
        tau: float = 1.0,
        target_net_update_freq: int = 200,  # ignored if tau != 1.0
        buffer_size: int = int(1e6),
        net_arch: List = [256, 256, 256, 256],
        batch_size: int = 256,
        learning_starts: int = 100,
        gradient_updates: int = 1,
        gamma: float = 0.99,
        max_grad_norm: Optional[float] = 1.0,
        envelope: bool = True,
        num_sample_w: int = 4,
        per: bool = True,
        per_alpha: float = 0.6,
        initial_homotopy_lambda: float = 0.0,
        final_homotopy_lambda: float = 1.0,
        homotopy_decay_steps: int = None,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "Envelope",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        group: Optional[str] = None,
        cnn_config: Optional[dict] = None,
        use_hv: bool = False,
        ref_point: np.ndarray = np.array([-100.0, -100.0]),
        dirichlet_alpha: float = 0.8,
        num_interps: int = 2,
        interp_weight: np.ndarray = np.array([0.5, 0.5]),
        ucb_beta: float = 1.0,
        ucb_n_candidates: int = 50,
        ucb_neighbor_dist: float = 0.2,
        ucb_window: int = 200,
        ucb_warmup_episodes: int = 100,
        ucb_epsilon: float = 0.2,
    ):
        """Envelope Q-learning algorithm.

        Args:
            env: The environment to learn from.
            learning_rate: The learning rate (alpha).
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            tau: The soft update coefficient (keep in [0, 1]).
            target_net_update_freq: The frequency with which the target network is updated.
            buffer_size: The size of the replay buffer.
            net_arch: The size of the hidden layers of the value net.
            batch_size: The size of the batch to sample from the replay buffer.
            learning_starts: The number of steps before learning starts i.e. the agent will be random until learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor (gamma).
            max_grad_norm: The maximum norm for the gradient clipping. If None, no gradient clipping is applied.
            envelope: Whether to use the envelope method.
            num_sample_w: The number of weight vectors to sample for the envelope target.
            per: Whether to use prioritized experience replay.
            per_alpha: The alpha parameter for prioritized experience replay.
            initial_homotopy_lambda: The initial value of the homotopy parameter for homotopy optimization.
            final_homotopy_lambda: The final value of the homotopy parameter.
            homotopy_decay_steps: The number of steps to decay the homotopy parameter over.
            project_name: The name of the project, for wandb logging.
            experiment_name: The name of the experiment, for wandb logging.
            wandb_entity: The entity of the project, for wandb logging.
            log: Whether to log to wandb.
            seed: The seed for the random number generator.
            device: The device to use for training.
            group: The wandb group to use for logging.
            ref_point: reference point for the hypervolume computation.
        """
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device)
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.per = per
        self.per_alpha = per_alpha
        self.gradient_updates = gradient_updates
        self.initial_homotopy_lambda = initial_homotopy_lambda
        self.final_homotopy_lambda = final_homotopy_lambda
        self.homotopy_decay_steps = homotopy_decay_steps
        self.use_hv = use_hv
        self.ref_point = ref_point
        self.dirichlet_alpha = dirichlet_alpha
        self.num_interps = num_interps
        self.interp_weight = interp_weight  # default / eval interp weight
        self.ucb_beta = ucb_beta
        self.ucb_n_candidates = ucb_n_candidates
        self.ucb_neighbor_dist = ucb_neighbor_dist
        self.ucb_window = ucb_window
        self.ucb_warmup_episodes = ucb_warmup_episodes
        self.ucb_epsilon = ucb_epsilon

        # The env emits a flattened per-interpretation reward matrix as a single
        # vector of length num_interps * net_reward_dim, laid out row-major (one
        # row per interpretation, each row a self-contained objective vector
        # [shared/task, ...]). The agent reshapes it back into rows and trains
        # network i on row i.
        self.flat_reward_dim = self.reward_dim  # reward_space.shape[0]
        assert self.flat_reward_dim % self.num_interps == 0, (
            f"reward_dim={self.flat_reward_dim} is not divisible by "
            f"num_interps={self.num_interps}."
        )
        assert len(self.interp_weight) == self.num_interps, (
            f"interp_weight has length {len(self.interp_weight)} but "
            f"num_interps={self.num_interps}."
        )
        self.net_reward_dim = self.flat_reward_dim // self.num_interps

        self.q_nets = QNetEnsemble()
        self.target_q_nets = QNetEnsemble()

        if len(self.observation_shape) == 1:
            for i in range(num_interps):
                self.q_nets.append(QNet(
                    self.observation_shape,
                    self.action_dim,
                    self.net_reward_dim,
                    net_arch=net_arch,
                ).to(self.device))
                self.target_q_nets.append(QNet(
                    self.observation_shape,
                    self.action_dim,
                    self.net_reward_dim,
                    net_arch=net_arch,
                ).to(self.device))
        elif len(self.observation_shape) > 1:  # use CNNQNet
            self.q_nets.append(CNNQNet(
                self.observation_shape,
                self.action_dim,
                self.net_reward_dim,
                net_arch=net_arch,
                cnn_config=cnn_config,
            ).to(self.device))
            self.target_q_nets.append(CNNQNet(
                self.observation_shape,
                self.action_dim,
                self.net_reward_dim,
                net_arch=net_arch,
                cnn_config=cnn_config,
            ).to(self.device))

        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
            for param in target_q_net.parameters():
                param.requires_grad = False


        self.q_optim = optim.Adam(self.q_nets.parameters(), lr=self.learning_rate)

        self.envelope = envelope
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
        self._episode_rewards: deque = deque(maxlen=100)
        self._episode_lengths: deque = deque(maxlen=100)
        self._episode_mo_returns: List[np.ndarray] = []  # combined (agent-space) MO returns
        # Per-interpretation fronts for sum-of-HV UCB signal.
        self._episode_mo_returns_per_interp: List[List[np.ndarray]] = [
            [] for _ in range(self.num_interps)
        ]
        # Sliding window of (joint_key, normalised_marginal_hv) where
        # joint_key = concat(w, interp_weight).
        self._weight_hv_history: deque = deque(maxlen=self.ucb_window)
        # Current episode's interp weight (set at reset, used by max_action / get_q_sets).
        self._episode_interp_weight: np.ndarray = np.array(self.interp_weight, dtype=np.float32)
        self._n_updates: int = 0
        self._last_loss: float = float("nan")
        self._start_time: float = 0.0
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.flat_reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape,
                1,
                rew_dim=self.flat_reward_dim,
                max_size=buffer_size,
                action_dtype=np.uint8,
            )

        self.log = log
        if log:
            self.setup_wandb(project_name, experiment_name, wandb_entity, group)

    @override
    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "use_envelope": self.envelope,
            "num_sample_w": self.num_sample_w,
            "net_arch": self.net_arch,
            "per": self.per,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "initial_homotopy_lambda": self.initial_homotopy_lambda,
            "final_homotopy_lambda": self.final_homotopy_lambda,
            "homotopy_decay_steps": self.homotopy_decay_steps,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def save(
        self,
        save_replay_buffer: bool = True,
        save_dir: str = "weights/",
        filename: Optional[str] = None,
    ):
        """Save the model and the replay buffer if specified.

        Args:
            save_replay_buffer: Whether to save the replay buffer too.
            save_dir: Directory to save the model.
            filename: filename to save the model.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params["q_nets_state_dict"] = self.q_nets.state_dict()
        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.experiment_name if filename is None else filename
        th.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path: str, load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.

        Args:
            path: Path to the model.
            load_replay_buffer: Whether to load the replay buffer too.
        """
        params = th.load(path, weights_only=False)
        self.q_nets.load_state_dict(params["q_nets_state_dict"])
        self.target_q_nets.load_state_dict(params["q_nets_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def __sample_batch_experiences(self):
        return self.replay_buffer.sample(
            self.batch_size, to_tensor=True, device=self.device
        )

    def _to_agent_objective(self, reward) -> np.ndarray:
        """Project an env reward matrix into the agent's net-objective space.

        The env reward is a matrix [num_interps, net_reward_dim] (one row per
        interpretation). The agent reasons in a single net_reward_dim objective
        space, collapsing the interpretation rows via a weighted average with
        ``self.interp_weight`` (the same weighting used to scalarize the per-interp
        Q-nets). Used for logging/eval, which run against net_reward_dim weights.
        (Training itself keeps the interpretations separate per net.)

        Args:
            reward: an env reward matrix of shape [num_interps, net_reward_dim],
                or its flattened form.

        Returns: an array of shape [net_reward_dim].
        """
        reward = np.asarray(reward, dtype=float).reshape(
            self.num_interps, self.net_reward_dim
        )
        return np.asarray(self.interp_weight, dtype=float) @ reward

    @th.no_grad()
    def _eval_front_point(
        self, eval_env: gym.Env, w: np.ndarray, rep: int
    ) -> np.ndarray:
        """Evaluate the greedy policy under net-objective weight ``w``.

        The env hands back a reward matrix; the standard ``eval_mo`` would try to
        accumulate it against the (net_reward_dim,) weight and break. Here we act
        with the net weight but accumulate the env reward projected into the agent's
        net-objective space via ``_to_agent_objective``, averaging the discounted
        return over ``rep`` episodes. Returns a [net_reward_dim] Pareto-front point.
        """
        disc_returns = []
        for _ in range(rep):
            obs, _ = eval_env.reset()
            done = False
            disc_return = np.zeros(self.net_reward_dim, dtype=np.float32)
            gamma = 1.0
            while not done:
                action = self.eval(obs, w)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                disc_return += gamma * self._to_agent_objective(reward).astype(
                    np.float32
                )
                gamma *= self.gamma
            disc_returns.append(disc_return)
        return np.mean(disc_returns, axis=0)

    @override
    def update(self, fixed_w: Optional[np.ndarray] = None):
        critic_losses = []
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

            if fixed_w is not None:
                sampled_w = th.tensor(fixed_w).float().to(self.device).unsqueeze(0)
            else:
                sampled_w = (
                    th.tensor(
                        random_weights(
                            self.net_reward_dim,
                            self.num_sample_w,
                            dist="dirichlet",
                            rng=self.np_random,
                            alpha=self.dirichlet_alpha,
                        )
                    )
                    .float()
                    .to(self.device)
                )
            w = sampled_w.repeat_interleave(
                b_obs.size(0), 0
            )  # repeat the weights for each sample
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                b_obs.repeat(self.num_sample_w, *(1 for _ in range(b_obs.dim() - 1))),
                b_actions.repeat(self.num_sample_w, 1),
                b_rewards.repeat(self.num_sample_w, 1),
                b_next_obs.repeat(
                    self.num_sample_w, *(1 for _ in range(b_next_obs.dim() - 1))
                ),
                b_dones.repeat(self.num_sample_w, 1),
            )

            # the buffer stores the flattened env reward matrix; unflatten it back
            # into per-interpretation rows so net i is trained on row i.
            b_rewards_per_net = b_rewards.view(
                b_rewards.size(0), self.num_interps, self.net_reward_dim
            ).permute(1, 0, 2)  # [num_interps, N, net_reward_dim]

            # update each q net separately
            with th.no_grad():
                if self.envelope:
                    targets = self.envelope_target(b_next_obs, w, sampled_w)
                else:
                    targets = self.ddqn_target(b_next_obs, w)
                target_q = b_rewards_per_net + (1 - b_dones) * self.gamma * targets

            q_values = self.q_nets(b_obs, w)  # [num_interps, N, action_dim, reward_dim]
            q_value = q_values.gather(
                2,
                b_actions.long()
                .reshape(1, -1, 1, 1)
                .expand(q_values.size(0), q_values.size(1), 1, q_values.size(3)),
            ).squeeze(2)  # [num_interps, N, reward_dim]

            critic_loss = F.mse_loss(q_value, target_q)

            if self.homotopy_lambda > 0:
                wQ = th.einsum("inr,nr->in", q_value, w)
                wTQ = th.einsum("inr,nr->in", target_q, w)
                auxiliary_loss = F.mse_loss(wQ, wTQ)
                critic_loss = (
                    1 - self.homotopy_lambda
                ) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.log and self.global_step % 100 == 0:
                wandb.log(
                    {
                        "losses/grad_norm": get_grad_norm(
                            self.q_nets.parameters()
                        ).item(),
                        "global_step": self.global_step,
                    },
                )
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.q_nets.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (
                    q_value[:, : len(b_inds)] - target_q[:, : len(b_inds)]
                ).detach()  # [num_interps, b_inds, reward_dim]
                priority = th.einsum("isr,sr->is", td_err, w[: len(b_inds)]).abs()
                # average the priority across the q-nets
                priority = priority.mean(dim=0)
                priority = priority.cpu().numpy().flatten()
                priority = (
                    priority + self.replay_buffer.min_priority
                ) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(
                self.q_nets.parameters(), self.target_q_nets.parameters(), self.tau
            )

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        if self.log and self.global_step % 100 == 0:
            wandb.log(
                {
                    "losses/critic_loss": np.mean(critic_losses),
                    "metrics/epsilon": self.epsilon,
                    "metrics/homotopy_lambda": self.homotopy_lambda,
                    "global_step": self.global_step,
                },
            )
            if self.per:
                wandb.log({"metrics/mean_priority": np.mean(priority)})

        self._last_loss = float(np.mean(critic_losses))
        self._n_updates += self.gradient_updates

    def _dump_logs(self) -> None:
        """Print training statistics in Stable Baselines3 format."""
        if not self._episode_rewards:
            return
        time_elapsed = max(time.time() - self._start_time, 1e-9)
        fps = int(self.global_step / time_elapsed)
        rows = [
            ("rollout/", ""),
            ("   ep_len_mean", f"{np.mean(self._episode_lengths):.4g}"),
            ("   ep_rew_mean", f"{np.mean(self._episode_rewards):.4g}"),
            ("   exploration_rate", f"{self.epsilon:.4g}"),
            ("time/", ""),
            ("   episodes", str(self.num_episodes)),
            ("   fps", str(fps)),
            ("   time_elapsed", str(int(time_elapsed))),
            ("   total_timesteps", str(self.global_step)),
            ("train/", ""),
            ("   learning_rate", f"{self.learning_rate:.4g}"),
            ("   loss", f"{self._last_loss:.4g}"),
            ("   n_updates", str(self._n_updates)),
        ]
        key_w = max(len(k) for k, _ in rows)
        val_w = max(len(v) for _, v in rows)
        width = key_w + val_w + 7
        sep = "-" * width
        print(sep)
        for key, val in rows:
            print(f"| {key:<{key_w}} | {val:>{val_w}} |")
        print(sep)

    @override
    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        obs = th.as_tensor(obs).float().to(self.device)
        w = th.as_tensor(w).float().to(self.device)
        return self.max_action(obs, w)

    def act(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Epsilon-greedily select an action given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: an integer representing the action to take.
        """
        if self.np_random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            # if self.use_hv:
            #     return self.hv_max_action(obs)
            # else:
            return self.max_action(obs, w)

    def get_q_sets(self, obs):
        """Compute the Q-set for a given observation, for all actions.

        Args:
            obs: current observation (array or tensor)

        Returns:
            A set of non-dominated Q vectors (tuples).
        """
        sampled_w = (
            th.tensor(
                random_weights(
                    self.net_reward_dim,
                    self.num_sample_w,
                    dist="dirichlet",
                    rng=self.np_random,
                    alpha=1.0,
                )
            )
            .float()
            .to(self.device)
        )
        obs_per_weight = th.stack([obs] * self.num_sample_w)
        q_set = self.q_nets(obs_per_weight, sampled_w)  # [num_interps, K, action_dim, reward_dim]
        # scalarize between interps with self.interp_weight
        q_set = th.einsum(
            "i,ikar->kar",
            th.tensor(self.interp_weight).float().to(self.device),
            q_set,
        )  # [K, action_dim, reward_dim]
        return q_set

    def score_hypervolume(self, obs: th.Tensor) -> np.ndarray:
        """Compute the action scores based upon the hypervolume metric.

        Args:
            obs: current observation.

        Returns: a score per action (shape: [action_dim]).
        """
        q_sets = self.get_q_sets(obs).cpu().numpy()  # [K, action_dim, reward_dim]
        scores = np.array(
            [
                hypervolume(
                    self.ref_point,
                    list(get_non_dominated(set(map(tuple, q_sets[:, a, :].tolist())))),
                )
                for a in range(self.action_dim)
            ]
        )
        return scores

    @th.no_grad()
    def hv_max_action(self, obs: th.Tensor) -> int:
        """
        Select the action with the highest hypervolume contribution given an observation.
        """
        return int(np.argmax(self.score_hypervolume(obs)))
    
    @th.no_grad()
    def hv_best_weight(self, obs: th.Tensor) -> np.ndarray:
        """Select the weight whose greedy action has the highest hypervolume contribution."""
        sampled_w = (
            th.tensor(
                random_weights(
                    self.net_reward_dim,
                    self.num_sample_w,
                    dist="dirichlet",
                    rng=self.np_random,
                    alpha=1.0,
                )
            )
            .float()
            .to(self.device)
        )
        obs = th.tensor(obs).float().to(self.device)
        obs_per_weight = th.stack([obs] * self.num_sample_w)
        q_set = self.q_nets(obs_per_weight, sampled_w)  # [num_interps, K, action_dim, reward_dim]
        # scalarize between interps with self.interp_weight
        q_set = th.einsum("i,ibar->bar", th.tensor(self.interp_weight).float().to(self.device), q_set)  # [K, action_dim, reward_dim]
        hv_contributions = np.array(
            [
                hypervolume(
                    self.ref_point,
                    list(get_non_dominated(set(map(tuple, q_set[k].cpu().numpy().tolist())))),
                )
                for k in range(self.num_sample_w)
            ]
        )
        best_k = int(np.argmin(hv_contributions))
        return sampled_w[best_k].cpu().numpy()
    
    # def hv_weight_action(self, obs: th.Tensor, sampled_w: np.ndarray) -> int:
    #     """Pick the weight that contributes most to HV, then act greedily under it."""
    #     tensor_w = th.tensor(sampled_w).float().to(self.device)
    #     obs_stack = th.stack([obs] * len(sampled_w))
    #     q_vals = self.q_net(obs_stack, tensor_w).cpu().numpy()  # [K, A, R]
        
    #     # For each weight, pick its greedy action
    #     scalarized = np.einsum("kr,kar->ka", sampled_w, q_vals)  # [K, A]
    #     best_actions = np.argmax(scalarized, axis=1)  # [K]
        
    #     # Pick the weight whose greedy action contributes most to HV of the current front
    #     # (or simply the one with highest marginal HV contribution)
    #     # ... compute HV contributions per weight candidate ...
    #     best_k = ...  # whichever weight + action has highest HV contribution
    #     return int(best_actions[best_k])

    @th.no_grad()
    def max_action(self, obs: th.Tensor, w: th.Tensor) -> int:
        """Select the action with the highest Q-value given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: the action with the highest Q-value.
        """
        q_values = self.q_nets(obs, w)  # [num_interps, B, action_dim, reward_dim]
        # scalarize over objectives with w
        scalarized_q_values = th.einsum("r,ibar->iba", w, q_values)  # [num_interps, B, action_dim]
        # scalarize between interps with self.interp_weight
        scalarized_q_values = th.einsum(
            "i,iba->ba",
            th.tensor(self.interp_weight).float().to(self.device),
            scalarized_q_values,
        )  # [B, action_dim]
        max_act = th.argmax(scalarized_q_values, dim=1)
        return max_act.detach().item()

    @th.no_grad()
    def envelope_target(
        self, obs: th.Tensor, w: th.Tensor, sampled_w: th.Tensor
    ) -> th.Tensor:
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
        # Num interps X Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_nets(next_obs, W).view(
            self.num_interps, obs.size(0), sampled_w.size(0), self.action_dim, self.net_reward_dim
        )
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = th.einsum("br,ibwar->ibwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = th.max(scalarized_next_q_values, dim=3)
        # Max weights in the envelope
        pref = th.argmax(max_q, dim=2)

        # MO Q-values evaluated on the target networks
        next_q_values_target = self.target_q_nets(next_obs, W).view(
            self.num_interps, obs.size(0), sampled_w.size(0), self.action_dim, self.net_reward_dim
        )

        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            3,
            ac.unsqueeze(3)
            .unsqueeze(4)
            .expand(
                next_q_values.size(0),
                next_q_values.size(1),
                next_q_values.size(2),
                1,
                next_q_values.size(4),
            ),
        ).squeeze(3)
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(
            2,
            pref.reshape(self.num_interps, -1, 1, 1).expand(
                max_next_q.size(0), max_next_q.size(1), 1, max_next_q.size(3)
            ),
        ).squeeze(2)
        return max_next_q  # [num_interps, N, reward_dim]

    @th.no_grad()
    def ddqn_target(self, obs: th.Tensor, w: th.Tensor) -> th.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        # Max action for each state, per q-net
        q_values = self.q_nets(obs, w)  # [num_interps, N, action_dim, reward_dim]
        scalarized_q_values = th.einsum("br,ibar->iba", w, q_values)
        max_acts = th.argmax(scalarized_q_values, dim=2)  # [num_interps, N]
        # Action evaluated with the target networks
        q_values_target = self.target_q_nets(obs, w)  # [num_interps, N, action_dim, reward_dim]
        q_values_target = q_values_target.gather(
            2,
            max_acts.long()
            .reshape(q_values_target.size(0), -1, 1, 1)
            .expand(q_values_target.size(0), q_values_target.size(1), 1, q_values_target.size(3)),
        ).squeeze(2)
        return q_values_target  # [num_interps, N, reward_dim]

    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        weight: Optional[np.ndarray] = None,
        total_episodes: Optional[int] = None,
        reset_num_timesteps: bool = True,
        eval_freq: int = 10000,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        reset_learning_starts: bool = False,
        verbose: bool = False,
        num_checkpoints: int = 3,
        out_dir: str = "results/reach_goal/pareto_front",
    ):
        """Train the agent.

        Args:
            total_timesteps: total number of timesteps to train for.
            eval_env: environment to use for evaluation. If None, it is ignored.
            known_pareto_front: known pareto front for the hypervolume computation.
            weight: weight vector. If None, it is randomly sampled every episode (as done in the paper).
            total_episodes: total number of episodes to train for. If None, it is ignored.
            reset_num_timesteps: whether to reset the number of timesteps. Useful when training multiple times.
            eval_freq: policy evaluation frequency (in number of steps).
            num_eval_weights_for_front: number of weights to sample for creating the pareto front when evaluating.
            num_eval_episodes_for_front: number of episodes to run when evaluating the policy.
            num_eval_weights_for_eval (int): Number of weights use when evaluating the Pareto front, e.g., for computing expected utility.
            reset_learning_starts: whether to reset the learning starts. Useful when training multiple times.
            verbose: whether to print the episode info.
        """
        if eval_env is not None:
            assert (
                self.ref_point is not None
            ), "Reference point must be provided for the hypervolume computation."
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": self.ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "weight": weight if weight is not None else None,
                    "total_episodes": total_episodes,
                    "reset_num_timesteps": reset_num_timesteps,
                    "eval_freq": eval_freq,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "reset_learning_starts": reset_learning_starts,
                }
            )
        run_id = f"small__{total_timesteps}__{wandb.run.id}"
        out_dir = Path(out_dir) / run_id

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes
        if reset_learning_starts:  # Resets epsilon-greedy exploration
            self.learning_starts = self.global_step
        self._start_time = time.time()

        num_episodes = 0
        eval_weights = equally_spaced_weights(
            self.net_reward_dim, n=num_eval_weights_for_front
        )
        obs, _ = self.env.reset()

        if weight is not None:
            w = np.array(weight)
        elif self.use_hv:
            w = self.hv_best_weight(obs)
        else:
            w = random_weights(
                self.net_reward_dim, 1, dist="dirichlet", rng=self.np_random, alpha=self.dirichlet_alpha
            )

        tensor_w = th.tensor(w).float().to(self.device)

        for _ in range(1, total_timesteps + 1):
            # checkpoint: save model
            if self.global_step % (total_timesteps // num_checkpoints) == 0:
                self.save(save_dir=str(out_dir), filename="model")
            if total_episodes is not None and num_episodes == total_episodes:
                break

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.act(th.as_tensor(obs).float().to(self.device), tensor_w)

            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)
            self.global_step += 1

            self.replay_buffer.add(obs, action, vec_reward, next_obs, terminated)
            if self.global_step >= self.learning_starts:
                self.update(
                    fixed_w=weight
                )  # weight is only assigned a value if training on a single weight

            if eval_env is not None and self.log and self.global_step % eval_freq == 0:
                current_front = [
                    self._eval_front_point(
                        eval_env, ew, num_eval_episodes_for_front
                    )
                    for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=current_front,
                    hv_ref_point=self.ref_point,
                    reward_dim=self.net_reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )

            if terminated or truncated:
                obs, _ = self.env.reset()
                num_episodes += 1
                self.num_episodes += 1

                if "episode" in info.keys():
                    ep_info = info["episode"]
                    # project the env return (shared + per-interpretation) into the
                    # agent's 2-dim objective space so it matches the 2-dim weights
                    mo_return = self._to_agent_objective(ep_info["r"])
                    self._episode_mo_returns.append(mo_return)
                    ep_rew = np.dot(w.flatten(), mo_return)
                    self._episode_rewards.append(float(ep_rew))
                    self._episode_lengths.append(int(ep_info["l"]))
                    if verbose:
                        self._dump_logs()

                if weight is None:
                    if self.use_hv:
                        w = self.hv_best_weight(obs)
                    else:
                        w = random_weights(
                            self.net_reward_dim, 1, dist="dirichlet", rng=self.np_random, alpha=self.dirichlet_alpha
                        )
                    tensor_w = th.tensor(w).float().to(self.device)

            else:
                obs = next_obs
