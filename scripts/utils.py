import gymnasium as gym
from mo_gymnasium.wrappers.wrappers import MORecordEpisodeStatistics

from agent.envelope import Envelope
from agent.pql import PQL

from pathlib import Path


AGENTS = {'envelope': Envelope, 'pql': PQL}

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
    env = gym.make(**env_config)
    env = MORecordEpisodeStatistics(env)
    return env