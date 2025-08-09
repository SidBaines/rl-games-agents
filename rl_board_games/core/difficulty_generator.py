#!/usr/bin/env python3
"""
Utility to generate difficulty lookup tables for Ricochet Robots curriculum.

This script pre-computes optimal solve lengths for different board configurations
to enable efficient difficulty-based curriculum learning.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from ..games.ricochet_robots.board import Board
from ..games.ricochet_robots.game import RicochetRobotsGame
from ..games.ricochet_robots.solver_astar import AStarSolver
from .curriculum import DifficultyLookup


class DifficultyGenerator:
    """Generates difficulty lookup tables for curriculum learning."""

    def __init__(self, lookup_dir: str | Path = "difficulty_lookup"):
        self.difficulty_lookup = DifficultyLookup(lookup_dir)
        self.solver_timeout = 30  # seconds

    def generate_lookup_table(
        self,
        board_size: int,
        num_robots: int,
        num_samples: int = 10000,
        max_walls_range: Tuple[int, int] = (5, 25),
        seed_offset: int = 0,
    ) -> Dict[int, int]:
        """Generate lookup table for given configuration.

        Returns a mapping from RNG seed -> optimal solve length (number of moves),
        for seeds where a solution within the configured depth/timeout was found.
        """

        print(f"Generating lookup table for {board_size}x{board_size} board with {num_robots} robots...")
        print(f"Target samples: {num_samples}")

        lookup_table = self.difficulty_lookup.load_lookup_table(board_size, num_robots)
        existing_count = len(lookup_table)
        print(f"Found {existing_count} existing entries")

        generated_count = 0
        failed_count = 0

        for i in range(num_samples):
            seed = seed_offset + i

            # Skip if already computed
            if seed in lookup_table:
                continue

            try:
                solve_length = self._compute_solve_length(
                    seed=seed,
                    board_size=board_size,
                    num_robots=num_robots,
                    max_walls_range=max_walls_range,
                )

                if solve_length is not None:
                    lookup_table[seed] = solve_length
                    generated_count += 1

                    # Periodic progress + save
                    if generated_count % 100 == 0:
                        print(f"Generated {generated_count} new entries...")
                        self.difficulty_lookup.save_lookup_table(board_size, num_robots, lookup_table)
                else:
                    failed_count += 1

            except Exception as e:
                print(f"Error processing seed {seed}: {e}")
                failed_count += 1

        # Final save
        self.difficulty_lookup.save_lookup_table(board_size, num_robots, lookup_table)

        print("\nGeneration complete:")
        print(f"  Total entries: {len(lookup_table)}")
        print(f"  New entries: {generated_count}")
        print(f"  Failed: {failed_count}")

        return lookup_table

    def _compute_solve_length(
        self,
        seed: int,
        board_size: int,
        num_robots: int,
        max_walls_range: Tuple[int, int],
    ) -> int | None:
        """Compute solve length for a specific seed."""
        rng = random.Random(seed)

        # Generate board with random walls
        num_walls = rng.randint(*max_walls_range)
        board = Board.random_walls(size=board_size, num_walls=num_walls, rng=rng)

        # Create game
        game = RicochetRobotsGame(board=board, num_robots=num_robots, rng=rng)

        # Reset with same seed for deterministic initial state
        state = game.reset(seed=seed)

        # Solve with A*
        solver = AStarSolver(game, max_depth=50)  # Increased depth for larger boards

        start_time = time.time()
        try:
            solution = solver.solve(state)
            solve_time = time.time() - start_time

            if solve_time > self.solver_timeout:
                print(f"  Timeout for seed {seed} (took {solve_time:.2f}s)")
                return None

            if solution:
                return len(solution)
            else:
                # No solution found within max_depth
                return None

        except Exception as e:
            print(f"  Solver error for seed {seed}: {e}")
            return None

    def analyze_difficulty_distribution(self, board_size: int, num_robots: int) -> Dict[str, any]:
        """Analyze difficulty distribution for a configuration."""
        lookup_table = self.difficulty_lookup.load_lookup_table(board_size, num_robots)

        if not lookup_table:
            print(f"No lookup table found for {board_size}x{board_size} with {num_robots} robots")
            return {}

        solve_lengths = list(lookup_table.values())
        solve_lengths_sorted = sorted(solve_lengths)

        analysis = {
            "total_puzzles": len(solve_lengths),
            "min_solve_length": min(solve_lengths),
            "max_solve_length": max(solve_lengths),
            "mean_solve_length": sum(solve_lengths) / len(solve_lengths),
            "median_solve_length": solve_lengths_sorted[len(solve_lengths_sorted) // 2],
        }

        # Distribution by difficulty ranges
        easy_count = sum(1 for length in solve_lengths if length <= 3)
        medium_count = sum(1 for length in solve_lengths if 4 <= length <= 6)
        hard_count = sum(1 for length in solve_lengths if 7 <= length <= 10)
        expert_count = sum(1 for length in solve_lengths if length > 10)

        analysis["difficulty_distribution"] = {
            "easy (1-3 moves)": easy_count,
            "medium (4-6 moves)": medium_count,
            "hard (7-10 moves)": hard_count,
            "expert (11+ moves)": expert_count,
        }

        return analysis

    def generate_standard_lookup_tables(self) -> None:
        """Generate standard lookup tables for common configurations."""
        configs = [
            (4, 2, 5000),  # 4x4 board, 2 robots
            (6, 2, 7500),  # 6x6 board, 2 robots
            (8, 3, 10000),  # 8x8 board, 3 robots
            (12, 4, 15000),  # 12x12 board, 4 robots
            (16, 4, 20000),  # 16x16 board, 4 robots
        ]

        for board_size, num_robots, num_samples in configs:
            print(f"\n{'=' * 60}")
            self.generate_lookup_table(board_size, num_robots, num_samples)
            analysis = self.analyze_difficulty_distribution(board_size, num_robots)

            print(f"\nAnalysis for {board_size}x{board_size} with {num_robots} robots:")
            for key, value in analysis.items():
                if key == "difficulty_distribution":
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Generate difficulty lookup tables")
    parser.add_argument("--board-size", type=int, help="Board size (square)")
    parser.add_argument("--num-robots", type=int, help="Number of robots")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--lookup-dir", type=str, default="difficulty_lookup", help="Directory for lookup tables")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing lookup table")
    parser.add_argument("--generate-standard", action="store_true", help="Generate standard lookup tables")

    args = parser.parse_args()

    generator = DifficultyGenerator(args.lookup_dir)

    if args.generate_standard:
        generator.generate_standard_lookup_tables()
    elif args.analyze:
        if not args.board_size or not args.num_robots:
            print("Error: --board-size and --num-robots required for analysis")
            return
        analysis = generator.analyze_difficulty_distribution(args.board_size, args.num_robots)
        print(f"Analysis for {args.board_size}x{args.board_size} with {args.num_robots} robots:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    else:
        if not args.board_size or not args.num_robots:
            print("Error: --board-size and --num-robots required")
            return
        generator.generate_lookup_table(args.board_size, args.num_robots, args.num_samples)


if __name__ == "__main__":
    main()