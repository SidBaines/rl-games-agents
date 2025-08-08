"""
Core abstractions shared across all games and agents.
"""
from .game import Game, GameState
from .agent import Agent
from .encoder import Encoder
from .curriculum import Curriculum
from .solver import HeuristicSolver
from .persistence import CheckpointManager

__all__ = [
    "Game",
    "GameState",
    "Agent",
    "Encoder",
    "Curriculum",
    "HeuristicSolver",
    "CheckpointManager",
] 