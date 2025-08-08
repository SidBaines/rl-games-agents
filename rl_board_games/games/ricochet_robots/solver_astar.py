from __future__ import annotations

import heapq
from typing import Any, Dict, List, Tuple

from ...core.solver import HeuristicSolver
from .game import RRGameState, RicochetRobotsGame, Action
from .board import NORTH, EAST, SOUTH, WEST


class AStarSolver(HeuristicSolver):
    """A* search for shortest solution in Ricochet Robots."""

    def __init__(self, game: RicochetRobotsGame, max_depth: int = 15):
        self.game = game
        self.max_depth = max_depth

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------
    def _heuristic(self, state: RRGameState) -> int:
        gx, gy = state.goal
        rx, ry = state.robots[state.goal_robot]
        return abs(gx - rx) + abs(gy - ry)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, state: RRGameState) -> List[Action]:
        start_state = state
        frontier: List[Tuple[int, int, RRGameState]] = []
        heapq.heappush(frontier, (self._heuristic(start_state), 0, start_state))
        came_from: Dict[RRGameState, Tuple[RRGameState, Action]] = {}
        g_cost: Dict[RRGameState, int] = {start_state: 0}
        counter = 0  # tie-breaker
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if g_cost[current] > self.max_depth:
                # Exceeded max depth, fail
                return []
            if current.is_terminal:
                return self._reconstruct_path(came_from, current)
            # Expand all legal actions
            for action in self.game.legal_actions(current):
                next_state, _, _, _ = self._step_cached(current, action)
                tentative_g = g_cost[current] + 1
                if next_state not in g_cost or tentative_g < g_cost[next_state]:
                    g_cost[next_state] = tentative_g
                    priority = tentative_g + self._heuristic(next_state)
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, next_state))
                    came_from[next_state] = (current, action)
        return []  # no solution found

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _step_cached(self, state: RRGameState, action: Action):
        # Re-implementing step without altering original game state for speed
        robot_idx, direction = action
        robots = list(state.robots)
        x, y = robots[robot_idx]
        nx, ny = self.game.board.next_position(x, y, direction, robots)
        robots[robot_idx] = (nx, ny)
        new_state = RRGameState(
            robots=tuple(robots),
            goal=state.goal,
            goal_robot=state.goal_robot,
            move_count=state.move_count + 1,
            board=state.board,
        )
        done = new_state.is_terminal
        reward = self.game.reward_goal if done else self.game.reward_per_move
        info: Dict[str, Any] = {}
        return new_state, reward, done, info

    def _reconstruct_path(
        self, came_from: Dict[RRGameState, Tuple[RRGameState, Action]], goal_state: RRGameState
    ) -> List[Action]:
        actions: List[Action] = []
        current = goal_state
        while current in came_from:
            prev_state, action = came_from[current]
            actions.append(action)
            current = prev_state
        actions.reverse()
        return actions 