from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from ...core.game import Game, GameState
from .board import Board, NORTH, EAST, SOUTH, WEST
from .rendering import render_rgb

Direction = int  # alias
Action = Tuple[int, Direction]  # (robot_index, direction)


@dataclass(frozen=True)
class RRGameState(GameState):
    """Immutable state for Ricochet Robots."""

    robots: Tuple[Tuple[int, int], ...]  # positions (x,y)
    goal: Tuple[int, int]
    goal_robot: int  # index of robot that must reach goal
    move_count: int
    # Full Board reference excluded from equality/hash to keep it lightweight
    board: Board = field(compare=False, hash=False, repr=False)
    # Compact signature that captures board walls+size for hashing/equality
    board_sig: int = 0

    def __post_init__(self):
        # Populate board_sig if not provided
        if self.board_sig == 0 and isinstance(self.board, Board):
            object.__setattr__(self, "board_sig", self.board.signature())

    @property
    def is_terminal(self) -> bool:
        return self.robots[self.goal_robot] == self.goal


class RicochetRobotsGame(Game):
    """
    Ricochet Robots single-target variant.

    Simplifications:
    - Fixed number of robots (default 4)
    - Episode terminates when *goal_robot* reaches goal square.
    - Reward: -1 per move, +100 on success (configurable).
    """

    def __init__(
        self,
        board: Board | None = None,
        num_robots: int = 4,
        reward_per_move: float = -1.0,
        reward_goal: float = 100.0,
        rng: random.Random | None = None,
    ) -> None:
        self.board = board or Board.empty(16)
        self.num_robots = num_robots
        self.reward_per_move = reward_per_move
        self.reward_goal = reward_goal
        self.rng = rng or random.Random()

        # internal state
        self._state: RRGameState | None = None

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> RRGameState:
        if seed is not None:
            self.rng.seed(seed)
        robots = self._random_robot_positions()
        goal_robot = self.rng.randrange(self.num_robots)
        goal = self._random_empty_cell(exclude=robots)
        self._state = RRGameState(
            robots=robots,
            goal=goal,
            goal_robot=goal_robot,
            move_count=0,
            board=self.board,
        )
        return self._state

    def step(self, action: Action) -> Tuple[RRGameState, float, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Game not reset")
        robot_idx, direction = action
        if not (0 <= robot_idx < self.num_robots):
            raise ValueError("invalid robot index")
        robots = list(self._state.robots)
        x, y = robots[robot_idx]
        nx, ny = self.board.next_position(x, y, direction, robots)
        # If position unchanged, we still count move but maybe penalize
        robots[robot_idx] = (nx, ny)
        new_state = RRGameState(
            robots=tuple(robots),
            goal=self._state.goal,
            goal_robot=self._state.goal_robot,
            move_count=self._state.move_count + 1,
            board=self.board,
        )
        self._state = new_state
        done = new_state.is_terminal
        reward = self.reward_goal if done else self.reward_per_move
        info: Dict[str, Any] = {}
        return new_state, reward, done, info

    def legal_actions(self, state: RRGameState | None = None) -> List[Action]:
        st = state or self._state
        if st is None:
            raise RuntimeError("Game not reset")
        actions: List[Action] = []
        robots = list(st.robots)
        for idx, (x, y) in enumerate(st.robots):
            for direction in (NORTH, EAST, SOUTH, WEST):
                nx, ny = self.board.next_position(x, y, direction, robots)
                if (nx, ny) != (x, y):
                    actions.append((idx, direction))
        return actions

    def render(self, state: RRGameState | None = None, mode: str = "human") -> Any:
        st = state or self._state
        if st is None:
            raise RuntimeError("Game not reset")
        if mode == "human":
            self._render_text(st)
        elif mode == "rgb_array":
            return self._render_array(st)
        else:
            raise ValueError(f"Unsupported render mode {mode}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_empty_cell(self, exclude: Tuple[Tuple[int, int], ...]) -> Tuple[int, int]:
        while True:
            x = self.rng.randrange(self.board.width)
            y = self.rng.randrange(self.board.height)
            if (x, y) not in exclude:
                return (x, y)

    def _random_robot_positions(self) -> Tuple[Tuple[int, int], ...]:
        positions: List[Tuple[int, int]] = []
        while len(positions) < self.num_robots:
            pos = (self.rng.randrange(self.board.width), self.rng.randrange(self.board.height))
            if pos not in positions:
                positions.append(pos)
        return tuple(positions)

    # ------------------------------------------------------------------
    # Text renderer for quick debugging
    # ------------------------------------------------------------------
    def _render_text(self, st: RRGameState) -> None:
        # Build ASCII board with walls. Each cell becomes " X " etc.
        horiz = "─"
        vert = "│"
        corner = "┼"

        def cell_char(x, y):
            # Determine content char
            if (x, y) == st.goal:
                return "G"
            for idx, pos in enumerate(st.robots):
                if pos == (x, y):
                    ch = chr(ord("A") + idx)
                    # highlight goal robot
                    if idx == st.goal_robot:
                        return ch.upper()
                    return ch.lower()
            return " "

        rows = []
        for y in range(self.board.height):
            # Top border for row
            top_line = []
            cell_line = []
            for x in range(self.board.width):
                top_line.append(corner)
                top_line.append(horiz if self.board.has_wall(x, y, NORTH) else " ")
                # cell content line segments
                cell_line.append(vert if self.board.has_wall(x, y, WEST) else " ")
                cell_line.append(cell_char(x, y))
            top_line.append(corner)
            cell_line.append(vert if self.board.has_wall(self.board.width - 1, y, EAST) else " ")
            rows.append("".join(top_line))
            rows.append("".join(cell_line))
        # Bottom border
        bottom = [corner]
        for x in range(self.board.width):
            bottom.append(horiz if self.board.has_wall(x, self.board.height - 1, SOUTH) else " ")
            bottom.append(corner)
        rows.append("".join(bottom))
        print("\n".join(rows))

    def _render_array(self, st: RRGameState) -> np.ndarray:
        # Use shared rendering helper for consistency with encoders
        return render_rgb(
            board=self.board,
            robots=st.robots,
            goal=st.goal,
            goal_robot=st.goal_robot,
            scale=20,
        ) 