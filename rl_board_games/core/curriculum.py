from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Any, Dict, List, Optional, Tuple


@dataclass
class CurriculumLevel:
    """Represents a curriculum difficulty level."""
    name: str
    min_solve_length: int
    max_solve_length: int
    success_threshold: float
    board_size: int
    num_robots: int
    max_walls: int
    episodes_per_evaluation: int = 100
    # Optional range for sampling board sizes to avoid overfitting to a single size
    board_size_min: Optional[int] = None
    board_size_max: Optional[int] = None
    # Optional per-level cap on episode length. If None, fall back to env default.
    max_episode_steps: Optional[int] = None


@dataclass
class CurriculumState:
    """Tracks current curriculum progress."""
    current_level: int
    episodes_completed: int
    success_rate: float
    total_episodes: int
    level_episodes: int
    
    def to_dict(self) -> Dict:
        return {
            "current_level": self.current_level,
            "episodes_completed": self.episodes_completed,
            "success_rate": self.success_rate,
            "total_episodes": self.total_episodes,
            "level_episodes": self.level_episodes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CurriculumState":
        return cls(
            current_level=data["current_level"],
            episodes_completed=data["episodes_completed"],
            success_rate=data["success_rate"],
            total_episodes=data["total_episodes"],
            level_episodes=data["level_episodes"]
        )


class DifficultyLookup:
    """Manages pre-computed difficulty lookup tables."""
    
    def __init__(self, lookup_dir: str | Path = "difficulty_lookup"):
        self.lookup_dir = Path(lookup_dir)
        self.lookup_dir.mkdir(exist_ok=True)
        self._cache: Dict[Tuple[int, int], Dict[int, int]] = {}
    
    def _get_filename(self, board_size: int, num_robots: int) -> Path:
        return self.lookup_dir / f"lookup_{board_size}x{board_size}_{num_robots}robots.json"
    
    def load_lookup_table(self, board_size: int, num_robots: int) -> Dict[int, int]:
        """Load lookup table for given configuration."""
        key = (board_size, num_robots)
        if key in self._cache:
            return self._cache[key]
        
        filename = self._get_filename(board_size, num_robots)
        if filename.exists():
            with open(filename, 'r') as f:
                data = json.load(f)
                # Convert string keys back to int
                lookup_table = {int(k): v for k, v in data.items()}
                self._cache[key] = lookup_table
                return lookup_table
        
        return {}
    
    def save_lookup_table(self, board_size: int, num_robots: int, lookup_table: Dict[int, int]):
        """Save lookup table for given configuration."""
        filename = self._get_filename(board_size, num_robots)
        with open(filename, 'w') as f:
            # Convert int keys to string for JSON serialization
            json_data = {str(k): v for k, v in lookup_table.items()}
            json.dump(json_data, f, indent=2)
        
        # Update cache
        self._cache[(board_size, num_robots)] = lookup_table
    
    def get_difficulty(self, seed: int, board_size: int, num_robots: int) -> Optional[int]:
        """Get solve length for given seed and configuration."""
        lookup_table = self.load_lookup_table(board_size, num_robots)
        return lookup_table.get(seed)
    
    def add_difficulty(self, seed: int, board_size: int, num_robots: int, solve_length: int):
        """Add difficulty entry to lookup table."""
        lookup_table = self.load_lookup_table(board_size, num_robots)
        lookup_table[seed] = solve_length
        self.save_lookup_table(board_size, num_robots, lookup_table)


class Curriculum(ABC):
    """
    Generates initial board configurations with (optional) progressive difficulty.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """
        Re-initialise the curriculum random generator.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        Yield successive 'initial_state' objects (type depends on the game).
        """
        raise NotImplementedError


class ProgressiveCurriculum(Curriculum):
    """
    Enhanced curriculum with difficulty progression based on agent performance.
    """
    
    def __init__(
        self,
        levels: List[CurriculumLevel],
        difficulty_lookup: Optional[DifficultyLookup] = None,
        evaluation_episodes: int = 100,
        rng: Optional[random.Random] = None
    ):
        self.levels = levels
        self.difficulty_lookup = difficulty_lookup or DifficultyLookup()
        self.evaluation_episodes = evaluation_episodes
        self.rng = rng or random.Random()
        
        # Initialize curriculum state
        self.state = CurriculumState(
            current_level=0,
            episodes_completed=0,
            success_rate=0.0,
            total_episodes=0,
            level_episodes=0
        )
        
        # Episode tracking for progression
        self._episode_results: List[bool] = []
        self._current_seeds: List[int] = []
    
    def reset(self, seed: int | None = None) -> None:
        """Reset curriculum state."""
        if seed is not None:
            self.rng.seed(seed)
        self.state = CurriculumState(
            current_level=0,
            episodes_completed=0,
            success_rate=0.0,
            total_episodes=0,
            level_episodes=0
        )
        self._episode_results = []
        self._current_seeds = []
    
    def record_episode_result(self, success: bool) -> None:
        """Record result of an episode for curriculum progression."""
        self._episode_results.append(success)
        self.state.episodes_completed += 1
        self.state.total_episodes += 1
        self.state.level_episodes += 1
        
        # Update success rate (rolling window)
        if len(self._episode_results) > self.evaluation_episodes:
            self._episode_results.pop(0)
        
        self.state.success_rate = sum(self._episode_results) / len(self._episode_results)
        
        # Check for level progression
        if self._should_advance_level():
            self._advance_level()
    
    def _should_advance_level(self) -> bool:
        """Check if agent should advance to next difficulty level."""
        if self.state.current_level >= len(self.levels) - 1:
            return False  # Already at max level
        
        current_level = self.levels[self.state.current_level]
        
        # Need enough episodes to evaluate
        if len(self._episode_results) < current_level.episodes_per_evaluation:
            return False
        
        # Check success rate threshold
        return self.state.success_rate >= current_level.success_threshold
    
    def _advance_level(self) -> None:
        """Advance to next difficulty level."""
        if self.state.current_level < len(self.levels) - 1:
            self.state.current_level += 1
            self.state.level_episodes = 0
            # Keep some episode history for stability
            keep_episodes = self.evaluation_episodes // 4
            self._episode_results = self._episode_results[-keep_episodes:]
            print(f"Curriculum advanced to level {self.state.current_level}: {self.levels[self.state.current_level].name}")
    
    def get_current_level(self) -> CurriculumLevel:
        """Get current difficulty level."""
        return self.levels[self.state.current_level]
    
    def get_state(self) -> CurriculumState:
        """Get current curriculum state."""
        return self.state
    
    def save_state(self, path: str | Path) -> None:
        """Save curriculum state to file."""
        with open(path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
    
    def load_state(self, path: str | Path) -> None:
        """Load curriculum state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.state = CurriculumState.from_dict(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get curriculum metrics for logging."""
        current_level = self.get_current_level()
        return {
            "curriculum/current_level": self.state.current_level,
            "curriculum/level_name": current_level.name,
            "curriculum/success_rate": self.state.success_rate,
            "curriculum/episodes_completed": self.state.episodes_completed,
            "curriculum/total_episodes": self.state.total_episodes,
            "curriculum/level_episodes": self.state.level_episodes,
            "curriculum/target_success_rate": current_level.success_threshold,
            "curriculum/current_board_size": current_level.board_size,
            "curriculum/current_num_robots": current_level.num_robots
        }
    
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Yield successive game instances based on current difficulty level."""
        raise NotImplementedError 