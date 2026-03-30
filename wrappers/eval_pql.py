import gymnasium as gym
from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
import numpy as np


class EvalPQLWrapper():
    """
    Wrapper that allows the evaluation of a PQL agent.

    """
    def __init__(self, env, agent: PQL):
        self.env = env
        self.pql = agent

    def eval(self, state: int, weight: np.ndarray):
        """Return the action for a given state and weight over the objectives."""
        state = int(np.ravel_multi_index(state, self.env.unwrapped.map.shape))
        pcs = self.pql.get_local_pcs(state)
        # get the action that maximizes the weighted sum of the objectives
        action = np.argmax(np.sum(np.array(pcs) * weight, axis=1))
        return action