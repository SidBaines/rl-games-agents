from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.game import Game
from ..core.encoder import Encoder


class GameEnv(gym.Env):
    """
    Gymnasium environment wrapper around our Game abstraction.
    """

    def __init__(self, game: Game, encoder: Encoder, max_episode_steps: int = 1000):
        super().__init__()
        self.game = game
        self.encoder = encoder
        self.max_episode_steps = max_episode_steps
        self._current_state = None
        self._step_count = 0

        # Define action space - we'll need to determine this from the game
        # For now, assume discrete actions (this should be game-specific)
        self._setup_action_space()
        self._setup_observation_space()

    def _setup_action_space(self):
        """Setup action space - this is game-specific."""
        # For Ricochet Robots: 4 directions × num_robots
        # We'll use a simple approach: flatten to single discrete space
        # This should be overridden for specific games
        self.action_space = spaces.Discrete(100)  # placeholder

    def _setup_observation_space(self):
        """Setup observation space based on encoder."""
        # Get a dummy state to determine observation shape
        dummy_state = self.game.reset(seed=0)
        dummy_obs = self.encoder.encode(dummy_state)
        
        if isinstance(dummy_obs, np.ndarray):
            if dummy_obs.ndim == 1:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=dummy_obs.dtype
                )
            elif dummy_obs.ndim == 3:  # CHW format
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=dummy_obs.shape, dtype=dummy_obs.dtype
                )
            else:
                raise ValueError(f"Unsupported observation shape: {dummy_obs.shape}")
        else:
            raise ValueError(f"Unsupported observation type: {type(dummy_obs)}")

    def reset(self, seed: int | None = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        self._current_state = self.game.reset(seed=seed)
        self._step_count = 0
        obs = self.encoder.encode(self._current_state)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self._current_state is None:
            raise RuntimeError("Environment not reset")

        # Convert flat action to game-specific action
        game_action = self._decode_action(action)
        
        # Take step in game
        self._current_state, reward, done, info = self.game.step(game_action)
        self._step_count += 1
        
        # Check for max episode steps
        truncated = self._step_count >= self.max_episode_steps
        
        obs = self.encoder.encode(self._current_state)
        return obs, reward, done, truncated, info

    def _decode_action(self, action: int) -> Any:
        """Convert flat action index to game-specific action format."""
        # This should be overridden for specific games
        # For Ricochet Robots: action = robot_idx * 4 + direction
        robot_idx = action // 4
        direction = action % 4
        return (robot_idx, direction)

    def render(self, mode: str = "human"):
        """Render the environment."""
        if self._current_state is None:
            return None
        return self.game.render(self._current_state, mode=mode) 