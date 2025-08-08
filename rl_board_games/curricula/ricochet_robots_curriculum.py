from __future__ import annotations

import random
from typing import Iterator, List, Optional

from ..core.curriculum import ProgressiveCurriculum, CurriculumLevel, DifficultyLookup
from ..games.ricochet_robots.board import Board
from ..games.ricochet_robots.game import RicochetRobotsGame


class RicochetRobotsCurriculum(ProgressiveCurriculum):
    """Curriculum for Ricochet Robots with difficulty-based progression."""
    
    def __init__(
        self,
        levels: Optional[List[CurriculumLevel]] = None,
        difficulty_lookup: Optional[DifficultyLookup] = None,
        evaluation_episodes: int = 100,
        fallback_wall_range: tuple[int, int] = (5, 25),
        max_fallback_attempts: int = 10,
        rng: Optional[random.Random] = None
    ):
        self.levels = levels or self._create_default_levels()
        self.fallback_wall_range = fallback_wall_range
        self.max_fallback_attempts = max_fallback_attempts
        
        super().__init__(
            levels=self.levels,
            difficulty_lookup=difficulty_lookup,
            evaluation_episodes=evaluation_episodes,
            rng=rng
        )
    
    def _create_default_levels(self) -> List[CurriculumLevel]:
        """Create default curriculum levels for Ricochet Robots."""
        return [
            CurriculumLevel(
                name="Easy",
                min_solve_length=1,
                max_solve_length=3,
                success_threshold=0.90,
                board_size=4,
                num_robots=2,
                max_walls=8,
                episodes_per_evaluation=50
            ),
            CurriculumLevel(
                name="Medium",
                min_solve_length=3,
                max_solve_length=6,
                success_threshold=0.80,
                board_size=6,
                num_robots=2,
                max_walls=12,
                episodes_per_evaluation=75
            ),
            CurriculumLevel(
                name="Hard",
                min_solve_length=5,
                max_solve_length=10,
                success_threshold=0.70,
                board_size=8,
                num_robots=3,
                max_walls=20,
                episodes_per_evaluation=100
            ),
            CurriculumLevel(
                name="Expert",
                min_solve_length=8,
                max_solve_length=15,
                success_threshold=0.60,
                board_size=12,
                num_robots=4,
                max_walls=30,
                episodes_per_evaluation=150
            ),
            CurriculumLevel(
                name="Master",
                min_solve_length=10,
                max_solve_length=20,
                success_threshold=0.50,
                board_size=16,
                num_robots=4,
                max_walls=40,
                episodes_per_evaluation=200
            )
        ]
    
    def __iter__(self) -> Iterator[RicochetRobotsGame]:
        """Generate games based on current difficulty level."""
        while True:
            current_level = self.get_current_level()
            game = self._generate_game_for_level(current_level)
            if game is not None:
                yield game
            else:
                # Fallback to random generation if lookup fails
                yield self._generate_fallback_game(current_level)
    
    def _generate_game_for_level(self, level: CurriculumLevel) -> RicochetRobotsGame | None:
        """Generate a game matching the specified difficulty level."""
        max_attempts = 1000
        
        for _ in range(max_attempts):
            seed = self.rng.randint(0, 1000000)
            
            # Check if we have difficulty info for this seed
            solve_length = self.difficulty_lookup.get_difficulty(
                seed, level.board_size, level.num_robots
            )
            
            if solve_length is not None:
                # Check if difficulty matches level requirements
                if level.min_solve_length <= solve_length <= level.max_solve_length:
                    return self._create_game_from_seed(seed, level)
            else:
                # No lookup info available, might need to generate it
                # For now, skip and try another seed
                continue
        
        return None  # Could not find suitable game
    
    def _generate_fallback_game(self, level: CurriculumLevel) -> RicochetRobotsGame:
        """Generate a fallback game when lookup fails."""
        for _ in range(self.max_fallback_attempts):
            seed = self.rng.randint(0, 1000000)
            game = self._create_game_from_seed(seed, level)
            
            # TODO: Could add heuristic difficulty estimation here
            # For now, just return the game
            return game
        
        # Last resort: completely random game
        return self._create_game_from_seed(self.rng.randint(0, 1000000), level)
    
    def _create_game_from_seed(self, seed: int, level: CurriculumLevel) -> RicochetRobotsGame:
        """Create a game from a seed with specified level parameters."""
        rng = random.Random(seed)
        
        # Generate board with appropriate wall density
        num_walls = min(level.max_walls, rng.randint(level.max_walls // 2, level.max_walls))
        board = Board.random_walls(size=level.board_size, num_walls=num_walls, rng=rng)
        
        # Create game
        game = RicochetRobotsGame(
            board=board,
            num_robots=level.num_robots,
            rng=rng
        )
        # Expose the seed that generated this game so that environments
        # can reset the game back to the exact state that was solved by the
        # curriculum difficulty lookup.  This guarantees that every episode
        # the board/robots/goal configuration remains solvable within the
        # desired number of steps.
        game.initial_seed = seed  # type: ignore[attr-defined]
        
        # Store seed for tracking
        self._current_seeds.append(seed)
        if len(self._current_seeds) > 1000:  # Limit memory usage
            self._current_seeds.pop(0)
        
        return game
    
    def get_seeds_for_level(self, level_name: str, count: int = 100) -> List[int]:
        """Get seeds that match a specific difficulty level."""
        target_level = None
        for level in self.levels:
            if level.name == level_name:
                target_level = level
                break
        
        if target_level is None:
            raise ValueError(f"Level '{level_name}' not found")
        
        seeds = []
        attempts = 0
        max_attempts = count * 10  # Reasonable upper bound
        
        while len(seeds) < count and attempts < max_attempts:
            seed = self.rng.randint(0, 1000000)
            attempts += 1
            
            solve_length = self.difficulty_lookup.get_difficulty(
                seed, target_level.board_size, target_level.num_robots
            )
            
            if solve_length is not None:
                if target_level.min_solve_length <= solve_length <= target_level.max_solve_length:
                    seeds.append(seed)
        
        return seeds
    
    def get_difficulty_statistics(self) -> dict:
        """Get statistics about difficulty distribution for current level."""
        current_level = self.get_current_level()
        
        # Sample difficulty distribution
        sample_size = 1000
        difficulties = []
        
        for _ in range(sample_size):
            seed = self.rng.randint(0, 1000000)
            solve_length = self.difficulty_lookup.get_difficulty(
                seed, current_level.board_size, current_level.num_robots
            )
            if solve_length is not None:
                difficulties.append(solve_length)
        
        if not difficulties:
            return {"error": "No difficulty data available"}
        
        # Calculate statistics
        difficulties.sort()
        n = len(difficulties)
        
        stats = {
            "level_name": current_level.name,
            "board_size": current_level.board_size,
            "num_robots": current_level.num_robots,
            "sample_size": n,
            "min_difficulty": min(difficulties),
            "max_difficulty": max(difficulties),
            "mean_difficulty": sum(difficulties) / n,
            "median_difficulty": difficulties[n // 2],
            "target_range": (current_level.min_solve_length, current_level.max_solve_length),
            "in_range_count": sum(1 for d in difficulties 
                                if current_level.min_solve_length <= d <= current_level.max_solve_length)
        }
        
        stats["in_range_percentage"] = (stats["in_range_count"] / n) * 100
        
        return stats