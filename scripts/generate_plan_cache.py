#!/usr/bin/env python3
"""
Generate A* plan cache entries for curriculum levels defined in a config YAML.

Usage:
    python scripts/generate_plan_cache.py configs/ricochet_robots/ppo_astar_curruculum.yaml \
        --min-per-level 5 --max-attempts-per-level 5000 --solver-max-depth 50

This script pre-populates the plan cache so training can sample diverse seeds per level
without re-solving on-the-fly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import random
from typing import List, Tuple

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.core.plan_cache import PlanDifficultyCache
from rl_board_games.curricula.astar_plan_curriculum import PlanCurriculumLevel
from rl_board_games.games.ricochet_robots.board import Board
from rl_board_games.games.ricochet_robots.game import RicochetRobotsGame
from rl_board_games.games.ricochet_robots.solver_astar import AStarSolver


def load_plan_levels_from_config(config: dict) -> Tuple[List[PlanCurriculumLevel], str]:
    """Return plan curriculum levels and the plan cache directory from YAML config.
    Falls back to defaults if levels are not specified.
    """
    curriculum_cfg = config["curriculum"]
    plan_cache_dir = curriculum_cfg.get("plan_cache_dir", "plan_lookup")

    # Build levels from YAML if provided
    levels: List[PlanCurriculumLevel] = []
    for level_cfg in curriculum_cfg.get("levels", []):
        # Only consider when curriculum type is astar_plan or when explicit plan fields present
        level = PlanCurriculumLevel(
            name=level_cfg["name"],
            min_solve_length=level_cfg["min_solve_length"],
            max_solve_length=level_cfg["max_solve_length"],
            success_threshold=level_cfg["success_threshold"],
            board_size=level_cfg["board_size"],
            num_robots=level_cfg["num_robots"],
            max_walls=level_cfg["max_walls"],
            episodes_per_evaluation=level_cfg["episodes_per_evaluation"],
            board_size_min=level_cfg.get("board_size_min"),
            board_size_max=level_cfg.get("board_size_max"),
            max_robots_moved=level_cfg.get("max_robots_moved", 1),
            min_robots_moved=level_cfg.get("min_robots_moved", 0),
            max_episode_steps=level_cfg.get("max_episode_steps"),
        )
        levels.append(level)

    # If YAML didn't provide levels, import defaults from AStarPlanCurriculum
    if not levels:
        from rl_board_games.curricula.astar_plan_curriculum import AStarPlanCurriculum
        defaults = AStarPlanCurriculum(levels=None).levels
        levels = list(defaults)

    return levels, plan_cache_dir


def choose_board_sizes(level: PlanCurriculumLevel) -> List[int]:
    if level.board_size_min is not None and level.board_size_max is not None:
        lo = int(level.board_size_min)
        hi = int(level.board_size_max)
        if lo > hi:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(level.board_size)]


def create_game_from_seed(seed: int, level: PlanCurriculumLevel, board_size: int) -> RicochetRobotsGame:
	rng = random.Random(seed)
	# For structured generation (size>=8 and even), do NOT draw an extra randint
	# before building the board, since Board.random_walls ignores num_walls there
	# and relies on rng state. For fallback small/odd sizes, keep randomized count.
	if int(board_size) >= 8 and int(board_size) % 2 == 0:
		board = Board.random_walls(size=int(board_size), num_walls=int(level.max_walls), rng=rng)
	else:
		max_walls = int(level.max_walls)
		num_walls = min(max_walls, rng.randint(max_walls // 2, max_walls))
		board = Board.random_walls(size=int(board_size), num_walls=int(num_walls), rng=rng)
	game = RicochetRobotsGame(board=board, num_robots=int(level.num_robots), rng=rng)
	return game


def extract_plan_features(plan) -> Tuple[int, int]:
    total_moves = len(plan)
    robots_moved = len({robot_idx for (robot_idx, _dir) in plan})
    return total_moves, robots_moved


def satisfies(level: PlanCurriculumLevel, total_moves: int, robots_moved: int) -> bool:
    if total_moves == 0 and robots_moved == 0:
        return False
    if total_moves < int(level.min_solve_length) or total_moves > int(level.max_solve_length):
        return False
    if robots_moved < int(level.min_robots_moved) or robots_moved > int(level.max_robots_moved):
        return False
    return True


def populate_cache_for_level(
    cache: PlanDifficultyCache,
    level: PlanCurriculumLevel,
    min_count: int,
    max_attempts: int,
    solver_max_depth: int,
    rng: random.Random,
) -> int:
    """Populate cache entries satisfying level constraints until at least min_count exist.
    Returns the number of matching cached seeds after the operation (across all board sizes).
    """
    total_found = 0
    for board_size in choose_board_sizes(level):
        print(f"Populating cache for board size {board_size}")
        # Count existing
        existing = cache.get_matching_seeds(
            board_size=board_size,
            num_robots=int(level.num_robots),
            predicate=lambda feats: (
                ("total_moves" in feats and "robots_moved" in feats)
                and feats.get("total_moves", 0) > 0
                and feats.get("robots_moved", 0) > 0
                and feats.get("total_moves", 10**9) <= int(level.max_solve_length)
                and feats.get("robots_moved", 10**9) <= int(level.max_robots_moved)
                and feats.get("robots_moved", -1) >= int(level.min_robots_moved)
                and int(level.min_solve_length) <= feats.get("total_moves", -1) <= int(level.max_solve_length)
            ),
        )
        needed = max(0, min_count - len(existing))
        if needed <= 0:
            total_found += len(existing)
            continue

        attempts = 0
        while attempts < max_attempts and needed > 0:
            print(f"Attempt {attempts} of {max_attempts}")
            attempts += 1
            seed = rng.randint(0, 1_000_000)
            game = create_game_from_seed(seed, level, board_size)
            state = game.reset(seed=seed)
            solver = AStarSolver(game, max_depth=int(solver_max_depth))
            try:
                print(f"Solving for seed {seed}")
                plan = solver.solve(state)
            except Exception:
                continue
            if plan is None:
                continue
            total_moves, robots_moved = extract_plan_features(plan)
            print(f"Seed {seed}: total_moves={total_moves}, robots_moved={robots_moved}")

            if 0:
                # Always record the solved plan in the cache to avoid wasted compute
                cache.add(
                    seed=seed,
                    board_size=int(board_size),
                    num_robots=int(level.num_robots),
                    total_moves=int(total_moves),
                    robots_moved=int(robots_moved),
                )
            else:
                # Record only non-zero plans in the cache to avoid 0-move pollution
                if total_moves > 0 and robots_moved > 0:
                    cache.add(
                        seed=seed,
                        board_size=int(board_size),
                        num_robots=int(level.num_robots),
                        total_moves=int(total_moves),
                        robots_moved=int(robots_moved),
                    )

            # Count towards the level only if constraints satisfied
            if satisfies(level, total_moves, robots_moved):
                needed -= 1
        # Recount for this size
        existing = cache.get_matching_seeds(
            board_size=board_size,
            num_robots=int(level.num_robots),
            predicate=lambda feats: (
                ("total_moves" in feats and "robots_moved" in feats)
                and feats.get("total_moves", 0) > 0
                and feats.get("robots_moved", 0) > 0
                and feats.get("total_moves", 10**9) <= int(level.max_solve_length)
                and feats.get("robots_moved", 10**9) <= int(level.max_robots_moved)
                and feats.get("robots_moved", -1) >= int(level.min_robots_moved)
                and int(level.min_solve_length) <= feats.get("total_moves", -1) <= int(level.max_solve_length)
            ),
        )
        total_found += len(existing)

    return total_found


def main():
    parser = argparse.ArgumentParser(description="Generate A* plan cache for curriculum levels")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--min-per-level", type=int, default=10, help="Minimum seeds per level to populate")
    parser.add_argument("--max-attempts-per-level", type=int, default=5000, help="Max attempts per level")
    parser.add_argument("--solver-max-depth", type=int, default=6, help="A* solver max depth")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    if config.get("curriculum", {}).get("type") != "astar_plan":
        print("Config curriculum.type is not 'astar_plan'; nothing to do.")
        sys.exit(0)

    levels, plan_cache_dir = load_plan_levels_from_config(config)
    cache = PlanDifficultyCache(plan_cache_dir)

    rng = random.Random(args.seed)

    print(f"Generating plan cache in '{plan_cache_dir}' for {len(levels)} levels...")
    all_ok = True
    for idx, level in enumerate(levels):
        found = populate_cache_for_level(
            cache=cache,
            level=level,
            min_count=args.min_per_level,
            max_attempts=args.max_attempts_per_level,
            solver_max_depth=args.solver_max_depth,
            rng=rng,
        )
        status = "OK" if found >= args.min_per_level else "INSUFFICIENT"
        print(f"  Level {idx} {level.name}: cached {found} seeds (target {args.min_per_level}) -> {status}")
        if found < args.min_per_level:
            all_ok = False

    if not all_ok:
        print("One or more levels have insufficient cached seeds.")
        sys.exit(2)

    print("Cache generation complete.")


if __name__ == "__main__":
    main() 