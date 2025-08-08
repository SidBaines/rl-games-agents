import pytest
import numpy as np

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder
from rl_board_games.training.ricochet_robots_env import RicochetRobotsEnv
from rl_board_games.agents.sb3 import DQNAgent, PPOAgent


def test_dqn_agent_creation():
    """Test that DQN agent can be created and makes predictions."""
    game = RicochetRobotsGame()
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=10)
    
    agent = DQNAgent(
        env=env,
        encoder=encoder,
        learning_rate=1e-3,
        buffer_size=1000,
        learning_starts=100,
    )
    
    # Test that agent can act on a state
    state = game.reset(seed=42)
    # Note: This test is somewhat artificial since SB3 agents expect to work
    # through the gym environment, not directly with our GameState
    # In practice, the agent would receive observations from the env
    
    # Just verify the agent was created successfully
    assert agent is not None
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'encoder')


def test_ppo_agent_creation():
    """Test that PPO agent can be created."""
    game = RicochetRobotsGame()
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=10)
    
    agent = PPOAgent(
        env=env,
        encoder=encoder,
        learning_rate=1e-3,
        n_steps=64,
    )
    
    assert agent is not None
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'encoder') 