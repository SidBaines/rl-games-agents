from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

# Direction indices
NORTH, EAST, SOUTH, WEST = range(4)
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]
# Bitmask encoding for walls: 1<<direction
DIR_MASKS = [1 << d for d in range(4)]
OPPOSITE_DIR = {NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST}


class Board:
    """Light-weight board with immutable walls and size."""

    def __init__(self, width: int = 16, height: int | None = None, walls: np.ndarray | None = None):
        self.width = int(width)
        self.height = int(height) if height is not None else int(width)
        if walls is None:
            self.walls = np.zeros((self.height, self.width), dtype=np.uint8)
            self._add_board_boundaries()
        else:
            assert walls.shape == (self.height, self.width)
            self.walls = walls.astype(np.uint8)

    # ---------------------------------------------------------------------
    # Wall helpers
    # ---------------------------------------------------------------------
    def _add_board_boundaries(self) -> None:
        # North & South outer walls
        self.walls[0, :] |= DIR_MASKS[NORTH]
        self.walls[-1, :] |= DIR_MASKS[SOUTH]
        # West & East outer walls
        self.walls[:, 0] |= DIR_MASKS[WEST]
        self.walls[:, -1] |= DIR_MASKS[EAST]

    def add_wall(self, x: int, y: int, direction: int) -> None:
        """Add a wall on cell (x,y) for the given direction and the opposite on the adjacent cell."""
        self.walls[y, x] |= DIR_MASKS[direction]
        nx, ny = x + DX[direction], y + DY[direction]
        if 0 <= nx < self.width and 0 <= ny < self.height:
            self.walls[ny, nx] |= DIR_MASKS[OPPOSITE_DIR[direction]]

    def has_wall(self, x: int, y: int, direction: int) -> bool:
        return bool(self.walls[y, x] & DIR_MASKS[direction])

    # ---------------------------------------------------------------------
    # Movement helpers
    # ---------------------------------------------------------------------
    def next_position(
        self,
        x: int,
        y: int,
        direction: int,
        robots: List[Tuple[int, int]] | Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, int]:
        """Return the position where a robot will stop given a direction."""
        width, height = self.width, self.height
        robot_set = set(robots)
        cx, cy = x, y
        while True:
            if self.has_wall(cx, cy, direction):
                break
            nx, ny = cx + DX[direction], cy + DY[direction]
            if not (0 <= nx < width and 0 <= ny < height):
                # Shouldn't happen due to boundary walls, but safe-guard.
                break
            if (nx, ny) in robot_set:
                break
            cx, cy = nx, ny
        return cx, cy

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def empty(cls, size: int = 16) -> "Board":
        return cls(width=size, height=size)

    @classmethod
    def random_walls(cls, size: int = 16, num_walls: int = 20, rng: random.Random | None = None) -> "Board":
        rng = rng or random.Random()
        board = cls.empty(size)
        for _ in range(num_walls):
            x = rng.randrange(1, size - 1)
            y = rng.randrange(1, size - 1)
            direction = rng.choice([NORTH, EAST, SOUTH, WEST])
            board.add_wall(x, y, direction)
        return board 