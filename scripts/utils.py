import gymnasium as gym
from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics

from agent.ecc_envelope import ECCEnvelope
from agent.envelope import Envelope
from agent.pql import PQL

from pathlib import Path

from agent.ucb_envelope import UCBEnvelope
from wrappers.matrix_to_vector import DeontologicalWrapper, UtilitarianWrapper


AGENTS = {'envelope': Envelope, 'pql': PQL, 'ucb_envelope': UCBEnvelope, 'ecc_envelope': ECCEnvelope}

def find_model_path(run_id):
    base = 'results/reach_goal'
    # any folder under base dir that contains run_id (recursive, multiple level of folders and i search the lowest level one)
    for path in Path(base).rglob(f'*{run_id}*'):
        if path.is_dir():
            return path / 'model.tar'


def make_agent(env, agent_config):
    if "algorithm" not in agent_config:
        agent_name = "envelope"
    else: 
        agent_name = agent_config.pop("algorithm")
    if agent_name not in AGENTS:
        raise ValueError(f"unknown agent name {agent_name}")
    return AGENTS[agent_name](env=env, **agent_config)


def make_env(env_config) -> gym.Env:
    use_util = env_config.pop("utilitarian_wrapper", False)
    use_deont = env_config.pop("deontological_wrapper", False)
    env = gym.make(**env_config)
    if use_util:
        env = UtilitarianWrapper(env)
    elif use_deont:
        env = DeontologicalWrapper(env)
    reward_wrapped = env  # reward space the agent actually sees
    env = MORecordEpisodeStatistics(env)
    # MORecordEpisodeStatistics infers its accumulator dim from env.unwrapped, which
    # bypasses reward wrappers. Sync it to the (possibly projected) reward space.
    if use_util or use_deont:
        rdim = reward_wrapped.reward_space.shape[0]
        env.reward_dim = rdim
        env.rewards_shape = (rdim,)
    return env