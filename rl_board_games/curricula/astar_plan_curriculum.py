from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Dict, Set, Tuple

from ..core.curriculum import ProgressiveCurriculum, CurriculumLevel
from ..games.ricochet_robots.board import Board
from ..games.ricochet_robots.game import RicochetRobotsGame, RRGameState
from ..games.ricochet_robots.solver_astar import AStarSolver
from ..core.plan_cache import PlanDifficultyCache


@dataclass
class PlanCurriculumLevel(CurriculumLevel):
    """Curriculum level constrained by A* plan properties.

    In addition to base fields, this level constrains the maximum total plan length
    and the maximum number of distinct robots used in the A* solution.
    """
    max_total_moves: int = 1
    max_robots_moved: int = 1


class AStarPlanCurriculum(ProgressiveCurriculum):
    """Curriculum that uses A* to classify board difficulty by plan constraints.

    - Generates candidate boards by sampling seeds and (optionally) board sizes per level
    - Solves with A* and accepts the board if the plan satisfies constraints
    - Exposes the seed via game.initial_seed to ensure deterministic resets
    """

    def __init__(
        self,
        levels: Optional[List[PlanCurriculumLevel]] = None,
        evaluation_episodes: int = 50,
        rng: Optional[random.Random] = None,
        max_attempts_per_level: int = 2000,
        solver_max_depth: int = 50,
        plan_cache_dir: str | None = "plan_lookup",
    ) -> None:
        self.levels: List[PlanCurriculumLevel] = levels or self._create_default_levels()
        self.max_attempts_per_level = max_attempts_per_level
        self.solver_max_depth = solver_max_depth
        self.plan_cache = PlanDifficultyCache(plan_cache_dir) if plan_cache_dir else None
        # Per-run cache of seeds already served per (board_size, num_robots, level_name)
        self._session_seen: Dict[Tuple[int, int, str], Set[int]] = {}
        super().__init__(levels=self.levels, difficulty_lookup=None, evaluation_episodes=evaluation_episodes, rng=rng)

    # Defaults derived from user examples; board size fixed initially but with ranges retained for flexibility
    def _create_default_levels(self) -> List[PlanCurriculumLevel]:
        return [
            PlanCurriculumLevel(
                name="R1-L1",
                min_solve_length=0,
                max_solve_length=1,
                success_threshold=0.90,
                board_size=8,
                num_robots=4,
                max_walls=16,
                episodes_per_evaluation=40,
                board_size_min=8,
                board_size_max=8,
                max_total_moves=1,
                max_robots_moved=1,
            ),
            PlanCurriculumLevel(
                name="R1-L3",
                min_solve_length=0,
                max_solve_length=3,
                success_threshold=0.85,
                board_size=8,
                num_robots=4,
                max_walls=20,
                episodes_per_evaluation=50,
                board_size_min=8,
                board_size_max=8,
                max_total_moves=3,
                max_robots_moved=1,
            ),
            PlanCurriculumLevel(
                name="R<=2-L3",
                min_solve_length=0,
                max_solve_length=3,
                success_threshold=0.80,
                board_size=8,
                num_robots=4,
                max_walls=20,
                episodes_per_evaluation=50,
                board_size_min=8,
                board_size_max=8,
                max_total_moves=3,
                max_robots_moved=2,
            ),
            PlanCurriculumLevel(
                name="R1-L5",
                min_solve_length=0,
                max_solve_length=5,
                success_threshold=0.75,
                board_size=10,
                num_robots=4,
                max_walls=24,
                episodes_per_evaluation=60,
                board_size_min=10,
                board_size_max=10,
                max_total_moves=5,
                max_robots_moved=1,
            ),
            PlanCurriculumLevel(
                name="R1-L10",
                min_solve_length=0,
                max_solve_length=10,
                success_threshold=0.70,
                board_size=10,
                num_robots=4,
                max_walls=28,
                episodes_per_evaluation=60,
                board_size_min=10,
                board_size_max=10,
                max_total_moves=10,
                max_robots_moved=1,
            ),
            PlanCurriculumLevel(
                name="R<=2-L5",
                min_solve_length=0,
                max_solve_length=5,
                success_threshold=0.65,
                board_size=12,
                num_robots=4,
                max_walls=32,
                episodes_per_evaluation=80,
                board_size_min=12,
                board_size_max=12,
                max_total_moves=5,
                max_robots_moved=2,
            ),
            PlanCurriculumLevel(
                name="R<=2-L10",
                min_solve_length=0,
                max_solve_length=10,
                success_threshold=0.60,
                board_size=12,
                num_robots=4,
                max_walls=36,
                episodes_per_evaluation=100,
                board_size_min=12,
                board_size_max=12,
                max_total_moves=10,
                max_robots_moved=2,
            ),
            PlanCurriculumLevel(
                name="R<=3-L7",
                min_solve_length=0,
                max_solve_length=7,
                success_threshold=0.55,
                board_size=14,
                num_robots=4,
                max_walls=40,
                episodes_per_evaluation=120,
                board_size_min=14,
                board_size_max=14,
                max_total_moves=7,
                max_robots_moved=3,
            ),
        ]

    def __iter__(self) -> Iterator[RicochetRobotsGame]:
        while True:
            level = self.get_current_level()
            game = self._generate_game_for_level(level)
            if game is not None:
                yield game
            else:
                # As a fallback, generate a random board without A* guarantee
                yield self._generate_fallback_game(level)

    # ----------------------- Generation helpers ----------------------- #
    def _choose_board_size(self, level: PlanCurriculumLevel) -> int:
        if level.board_size_min is not None and level.board_size_max is not None:
            return self.rng.randint(level.board_size_min, level.board_size_max)
        return level.board_size

    def _generate_game_for_level(self, level: PlanCurriculumLevel) -> Optional[RicochetRobotsGame]:
        board_size = self._choose_board_size(level)

        # 1) If cache exists, try sampling a matching cached seed first
        if self.plan_cache is not None:
            matching = self.plan_cache.get_matching_seeds(
                board_size=board_size,
                num_robots=level.num_robots,
                predicate=lambda feats: (
                    feats.get("total_moves", 10**9) <= level.max_total_moves
                    and feats.get("robots_moved", 10**9) <= level.max_robots_moved
                    and level.min_solve_length <= feats.get("total_moves", 10**9) <= level.max_solve_length
                ),
            )
            if matching:
                key = (board_size, level.num_robots, level.name)
                seen = self._session_seen.get(key, set())
                # Prefer unseen seeds this run
                candidates = [s for s in matching if s not in seen]
                if not candidates:
                    candidates = matching
                seed = self.rng.choice(candidates)
                # Mark seen for this run
                if key not in self._session_seen:
                    self._session_seen[key] = set()
                self._session_seen[key].add(seed)
                return self._create_game_from_seed(seed, level, board_size)

        # 2) Otherwise, sample seeds and solve; record plan features in cache as we go
        attempts = 0
        while attempts < self.max_attempts_per_level:
            attempts += 1
            seed = self.rng.randint(0, 1_000_000)
            game = self._create_game_from_seed(seed, level, board_size)
            # Ensure deterministic reset back to this state
            state = game.reset(seed=seed)
            plan = self._solve_with_timeout(game, state)
            if plan is None:
                # Cache negative info? We skip to avoid bloating, only store positives
                continue
            total_moves, robots_moved = self._extract_plan_features(plan)
            if self.plan_cache is not None:
                self.plan_cache.add(
                    seed=seed,
                    board_size=board_size,
                    num_robots=level.num_robots,
                    total_moves=total_moves,
                    robots_moved=robots_moved,
                )
            if self._plan_satisfies_features(total_moves, robots_moved, level):
                # Mark seen for this run
                key = (board_size, level.num_robots, level.name)
                if key not in self._session_seen:
                    self._session_seen[key] = set()
                self._session_seen[key].add(seed)
                return game
        return None

    def _generate_fallback_game(self, level: PlanCurriculumLevel) -> RicochetRobotsGame:
        # No guarantees about plan constraints
        board_size = self._choose_board_size(level)
        seed = self.rng.randint(0, 1_000_000)
        return self._create_game_from_seed(seed, level, board_size)

    def _create_game_from_seed(self, seed: int, level: PlanCurriculumLevel, board_size: int) -> RicochetRobotsGame:
        rng = random.Random(seed)
        num_walls = min(level.max_walls, rng.randint(level.max_walls // 2, level.max_walls))
        board = Board.random_walls(size=board_size, num_walls=num_walls, rng=rng)
        game = RicochetRobotsGame(board=board, num_robots=level.num_robots, rng=rng)
        game.initial_seed = seed  # type: ignore[attr-defined]
        return game

    def _solve_with_timeout(self, game: RicochetRobotsGame, state: RRGameState):
        # Simple solve without explicit wall-clock timeout; rely on depth bounds
        solver = AStarSolver(game, max_depth=self.solver_max_depth)
        try:
            return solver.solve(state)
        except Exception:
            return None

    def _extract_plan_features(self, plan) -> tuple[int, int]:
        """Return (total_moves, robots_moved) from a plan list[(robot_idx, dir)]."""
        total_moves = len(plan)
        robots_moved = len({robot_idx for (robot_idx, _dir) in plan})
        return total_moves, robots_moved

    def _plan_satisfies_features(self, total_moves: int, robots_moved: int, level: PlanCurriculumLevel) -> bool:
        if total_moves > level.max_total_moves:
            return False
        if robots_moved > level.max_robots_moved:
            return False
        if total_moves < level.min_solve_length or total_moves > level.max_solve_length:
            return False
        return True 