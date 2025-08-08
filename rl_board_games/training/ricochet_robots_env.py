from __future__ import annotations

from typing import Any

from gymnasium import spaces

from ..games.ricochet_robots import RicochetRobotsGame
from ..core.encoder import Encoder
from .gym_env import GameEnv


class RicochetRobotsEnv(GameEnv):
    """
    Gymnasium environment specifically for Ricochet Robots.
    """

    def __init__(self, game: RicochetRobotsGame, encoder: Encoder, max_episode_steps: int = 100):
        self.ricochet_game = game  # Store typed reference
        super().__init__(game, encoder, max_episode_steps)

    def _setup_action_space(self):
        """Setup action space for Ricochet Robots: 4 directions × num_robots."""
        num_actions = self.ricochet_game.num_robots * 4
        self.action_space = spaces.Discrete(num_actions)

    def _decode_action(self, action: int) -> Any:
        """Convert flat action to (robot_idx, direction) tuple."""
        robot_idx = action // 4
        direction = action % 4
        return (robot_idx, direction)

    def _encode_action(self, robot_idx: int, direction: int) -> int:
        """Convert (robot_idx, direction) to flat action index."""
        return robot_idx * 4 + direction 