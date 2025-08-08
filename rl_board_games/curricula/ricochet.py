from __future__ import annotations

import itertools
import random
from typing import Iterator, Any, List

from ..core.curriculum import Curriculum
from ..games.ricochet_robots.board import Board
from ..games.ricochet_robots.game import RicochetRobotsGame, RRGameState


class RandomBoardCurriculum(Curriculum):
    """Yield endless RicochetRobotsGame objects with random boards.

    Parameters
    ----------
    size : int
        Board size (square).
    wall_range : tuple[int, int]
        Min/max random walls to add each board.
    num_robots : int
        Number of robots per game.
    rng : random.Random | None
        Random generator for reproducibility.
    """

    def __init__(
        self,
        size: int = 16,
        wall_range: tuple[int, int] = (10, 25),
        num_robots: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self.size = size
        self.wall_range = wall_range
        self.num_robots = num_robots
        self.rng = rng or random.Random()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def __iter__(self) -> Iterator[RicochetRobotsGame]:
        while True:
            num_walls = self.rng.randint(*self.wall_range)
            board = Board.random_walls(size=self.size, num_walls=num_walls, rng=self.rng)
            yield RicochetRobotsGame(board=board, num_robots=self.num_robots, rng=self.rng)


class EasyThenHardCurriculum(Curriculum):
    """Start with small boards then gradually increase difficulty."""

    def __init__(
        self,
        sizes: List[int] | None = None,
        walls_per_size: int | None = None,
        episodes_per_stage: int = 1000,
        num_robots: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        self.sizes = sizes or [5, 8, 12, 16]
        self.walls_per_size = walls_per_size or 10
        self.episodes_per_stage = episodes_per_stage
        self.num_robots = num_robots
        self.rng = rng or random.Random()
        self._generator = self._make_generator()

    def _make_generator(self):
        for size in self.sizes:
            for _ in range(self.episodes_per_stage):
                board = Board.random_walls(size=size, num_walls=self.walls_per_size, rng=self.rng)
                yield RicochetRobotsGame(board=board, num_robots=self.num_robots, rng=self.rng)
        # After finishing sizes list, repeat hardest level indefinitely
        hardest_size = self.sizes[-1]
        while True:
            board = Board.random_walls(size=hardest_size, num_walls=self.walls_per_size, rng=self.rng)
            yield RicochetRobotsGame(board=board, num_robots=self.num_robots, rng=self.rng)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self._generator = self._make_generator()

    def __iter__(self):
        return self._generator 