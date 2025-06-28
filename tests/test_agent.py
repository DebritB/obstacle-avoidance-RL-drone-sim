import pytest
from src.environment import ContinuousActionDroneEnv
from src.agent import PPOAgent

def test_environment_reset():
    env = ContinuousActionDroneEnv(use_gui=False)
    obs = env.reset()
    assert len(obs) == 4  # x, y, z, distance to goal

def test_agent_action():
    env = ContinuousActionDroneEnv(use_gui=False)
    agent = PPOAgent(env)
    obs = env.reset()
    action, log_prob = agent.select_action(obs)
    assert len(action) == 3
    assert isinstance(log_prob, float) 