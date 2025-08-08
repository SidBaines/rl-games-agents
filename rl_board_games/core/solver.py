from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from .game import GameState


class HeuristicSolver(ABC):
    """
    Classical search / heuristic algorithm that can optionally act as an agent
    or provide expert demonstrations.
    """

    @abstractmethod
    def solve(self, state: GameState) -> List[Any]:
        """
        Return a sequence of actions that (ideally) reaches a terminal state.
        """
        raise NotImplementedError 