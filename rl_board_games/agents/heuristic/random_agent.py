from __future__ import annotations

import random
from typing import Any

from ...core.agent import Agent
from ...core.game import GameState, Game


class RandomAgent(Agent):
    """Agent that chooses random legal actions."""

    def __init__(self, game: Game, rng: random.Random | None = None):
        self.game = game
        self.rng = rng or random.Random()

    def act(self, state: GameState) -> Any:
        """Choose a random legal action."""
        legal_actions = self.game.legal_actions(state)
        if not legal_actions:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal_actions) 