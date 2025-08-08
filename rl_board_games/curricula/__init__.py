"""Curricula for generating sequences of board configurations or game states."""
from .ricochet import RandomBoardCurriculum, EasyThenHardCurriculum

__all__ = [
    "RandomBoardCurriculum",
    "EasyThenHardCurriculum",
] 