from __future__ import annotations

from typing import Any, List

from ...core.agent import Agent
from ...core.game import GameState
from ...core.solver import HeuristicSolver


class AStarAgent(Agent):
    """Agent that uses A* solver to choose actions."""

    def __init__(self, solver: HeuristicSolver):
        self.solver = solver
        self._cached_plan: List[Any] = []
        self._plan_state: GameState | None = None

    def act(self, state: GameState) -> Any:
        """
        Get the next action from A* plan.
        Re-plans if the state has changed from expected.
        """
        # If we have a cached plan and the state matches expectation, use next action
        if (
            self._cached_plan
            and self._plan_state is not None
            and state == self._plan_state
        ):
            return self._cached_plan.pop(0)

        # Otherwise, compute new plan
        self._cached_plan = self.solver.solve(state)
        if not self._cached_plan:
            raise ValueError("A* solver found no solution")

        # We don't know the exact next state, so we'll re-plan on next call
        self._plan_state = None
        return self._cached_plan.pop(0) 