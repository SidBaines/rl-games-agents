from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .game import GameState


class Agent(ABC):
    """
    Base class for all agents (learned or heuristic).
    """

    @abstractmethod
    def act(self, state: GameState) -> Any:
        """
        Choose an action given the current GameState.
        """
        raise NotImplementedError

    def learn(self, *args, **kwargs) -> None:
        """
        Optional training step; default is a no-op for non-learning agents.
        """

    def save(self, path: str | Path) -> None:
        """
        Persist parameters to disk.
        Default implementation does nothing.
        """

    def load(self, path: str | Path) -> None:
        """
        Load parameters from disk.
        Default implementation does nothing.
        """