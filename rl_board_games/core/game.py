from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List


@dataclass(frozen=True)
class GameState(ABC):
    """
    Immutable representation of a game state.

    Sub-classes should add concrete fields (e.g. board tensor, current player)
    and MAY override __hash__ for faster performance.
    """

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the position is a finished episode."""
        raise NotImplementedError


class Game(ABC):
    """
    A richer alternative to the OpenAI Gymnasium Env interface.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> GameState:
        """
        Start a new episode and return the initial state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: Any
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Apply action and return (next_state, reward, done, info).

        The signature mimics Gym for compatibility.
        """
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self, state: GameState) -> List[Any]:
        """
        Return a list of legal actions for the given GameState.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state: GameState | None = None, mode: str = "human") -> Any:
        """
        Render the current or provided state.

        mode:
            'rgb_array' → np.ndarray
            'human'     → print / pygame GUI
        """
        raise NotImplementedError 