from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ...core.encoder import Encoder
from .game import RRGameState
from .board import Board, NORTH, EAST, SOUTH, WEST, DIR_MASKS
from .rendering import render_rgb


class PlanarEncoder(Encoder):
    """
    Encode board as multi-channel (C, H, W) NumPy array suitable for CNNs.

    Channels:
    0..R-1 : one-hot plane per robot
    R      : goal plane
    R+1..R+4 : walls N,E,S,W planes (binary)
    """

    def encode(self, state: RRGameState) -> np.ndarray:
        h, w = state.board.height, state.board.width
        num_robots = len(state.robots)
        channels = num_robots + 1 + 4
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


class WallAwarePlanarEncoder(Encoder):
    """Planar encoder that includes wall channels given a Board reference."""
    # def __init__(self, board: Board):
    #     self.board = board

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


class RGBArrayEncoder(Encoder):
    """
    Encode the game state as an RGB image array suitable for CNNs.

    Returns a float32 array with shape (3, H, W) in [0, 1]. The rendering
    is visually similar to the game's rgb_array render:
    - walls: black lines
    - robots: solid colored squares
    - goal: colored border matching the goal robot

    The output resolution is (board.height * scale, board.width * scale).
    """

    def __init__(self, scale: int = 20):
        # scale = pixels per cell; must be >= 8 to draw borders nicely
        self.scale = max(8, int(scale))

    def encode(self, state: RRGameState) -> np.ndarray:
        # Use shared renderer (HWC uint8) then normalize/transposed to CHW float32
        img_hwc = render_rgb(
            board=state.board,
            robots=state.robots,
            goal=state.goal,
            goal_robot=state.goal_robot,
            scale=self.scale,
        )
        return (img_hwc.astype(np.float32) / 255.0).transpose(2, 0, 1) 