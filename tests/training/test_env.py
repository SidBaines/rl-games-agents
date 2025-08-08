import pytest
import numpy as np

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder
from rl_board_games.training.ricochet_robots_env import RicochetRobotsEnv


def test_ricochet_robots_env_creation():
    """Test that the Ricochet Robots environment can be created."""
    game = RicochetRobotsGame()
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=50)
    
    assert env.action_space.n == game.num_robots * 4  # 4 directions per robot
    assert env.observation_space is not None


def test_env_reset_and_step():
    """Test basic environment functionality."""
    game = RicochetRobotsGame()
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=10)
    
    # Test reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    
    # Test step
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_encoding_decoding():
    """Test action space conversion."""
    game = RicochetRobotsGame(num_robots=4)
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=10)
    
    # Test that we can encode and decode actions
    robot_idx, direction = 1, 2
    flat_action = env._encode_action(robot_idx, direction)
    decoded_robot_idx, decoded_direction = env._decode_action(flat_action)
    
    assert decoded_robot_idx == robot_idx
    assert decoded_direction == direction 