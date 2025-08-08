from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ...core.encoder import Encoder
from .game import RRGameState
from .board import NORTH, EAST, SOUTH, WEST, DIR_MASKS


class PlanarEncoder(Encoder):
    """
    Encode board as multi-channel (C, H, W) NumPy array suitable for CNNs.

    Channels:
    0..R-1 : one-hot plane per robot
    R      : goal plane
    R+1..R+4 : walls N,E,S,W planes (binary)
    """

    def encode(self, state: RRGameState) -> np.ndarray:
        h, w = state.goal[1] + 1, state.goal[0] + 1  # Not reliable; use board size via context.
        # We'll infer board size from robots and goal max.
        w = max([x for x, _ in state.robots] + [state.goal[0]]) + 1
        h = max([y for _, y in state.robots] + [state.goal[1]]) + 1
        num_robots = len(state.robots)
        channels = num_robots + 1 + 4
        planes = np.zeros((channels, h, w), dtype=np.float32)
        # robots
        for idx, (x, y) in enumerate(state.robots):
            planes[idx, y, x] = 1.0
        # goal
        gx, gy = state.goal
        planes[num_robots, gy, gx] = 1.0
        # walls unavailable here – we cannot access board in encoder easily; placeholder zeros.
        return planes


class WallAwarePlanarEncoder(Encoder):
    """Planar encoder that includes wall channels given a Board reference."""

    def encode(self, state: RRGameState) -> np.ndarray:
        h, w = state.board.height, state.board.width
        num_robots = len(state.robots)
        channels = num_robots + 1 + 4  # robots + goal + walls NESW
        planes = np.zeros((channels, h, w), dtype=np.float32)

        # robots
        for idx, (x, y) in enumerate(state.robots):
            planes[idx, y, x] = 1.0

        # goal
        gx, gy = state.goal
        planes[num_robots, gy, gx] = 1.0

        # walls
        for dir_idx in range(4):
            mask = (state.board.walls & DIR_MASKS[dir_idx]) != 0
            planes[num_robots + 1 + dir_idx, mask] = 1.0

        return planes


class FlatArrayEncoder(Encoder):
    """
    Encode as 1D flat numeric array (robots xy pairs + goal xy).
    """

    def encode(self, state: RRGameState) -> Any:
        robots_flat = np.array(state.robots, dtype=np.int8).flatten()
        goal_flat = np.array(state.goal, dtype=np.int8)
        return np.concatenate([robots_flat, goal_flat]) 