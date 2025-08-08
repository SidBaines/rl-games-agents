from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .game import GameState


class Encoder(ABC):
    """
    Converts a domain-specific GameState into a tensor or other observation
    suitable for an RL agent.
    """

    @abstractmethod
    def encode(self, state: GameState) -> Any:
        """
        Return observation; shape / dtype depends on concrete encoder.
        """
        raise NotImplementedError 