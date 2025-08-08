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
        seed_offset: int = 0
    ) -> Dict[int, int]:
        """Generate lookup table for given configuration."""
        
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
                    seed, board_size, num_robots, max_walls_range
                )\
                \n                if solve_length is not None:\n                    lookup_table[seed] = solve_length\n                    generated_count += 1\n                    \n                    if generated_count % 100 == 0:\n                        print(f\"Generated {generated_count} new entries...\")\n                        # Save periodically\n                        self.difficulty_lookup.save_lookup_table(board_size, num_robots, lookup_table)\n                else:\n                    failed_count += 1\n                    \n            except Exception as e:\n                print(f\"Error processing seed {seed}: {e}\")\n                failed_count += 1\n        \n        # Final save\n        self.difficulty_lookup.save_lookup_table(board_size, num_robots, lookup_table)\n        \n        print(f\"\\nGeneration complete:\")\n        print(f\"  Total entries: {len(lookup_table)}\")\n        print(f\"  New entries: {generated_count}\")\n        print(f\"  Failed: {failed_count}\")\n        \n        return lookup_table\n    \n    def _compute_solve_length(self, seed: int, board_size: int, num_robots: int, max_walls_range: Tuple[int, int]) -> int | None:\n        \"\"\"Compute solve length for a specific seed.\"\"\"\n        rng = random.Random(seed)\n        \n        # Generate board with random walls\n        num_walls = rng.randint(*max_walls_range)\n        board = Board.random_walls(size=board_size, num_walls=num_walls, rng=rng)\n        \n        # Create game\n        game = RicochetRobotsGame(board=board, num_robots=num_robots, rng=rng)\n        \n        # Reset with same seed for deterministic initial state\n        state = game.reset(seed=seed)\n        \n        # Solve with A*\n        solver = AStarSolver(game, max_depth=50)  # Increased depth for larger boards\n        \n        start_time = time.time()\n        try:\n            solution = solver.solve(state)\n            solve_time = time.time() - start_time\n            \n            if solve_time > self.solver_timeout:\n                print(f\"  Timeout for seed {seed} (took {solve_time:.2f}s)\")\n                return None\n            \n            if solution:\n                return len(solution)\n            else:\n                # No solution found within max_depth\n                return None\n                \n        except Exception as e:\n            print(f\"  Solver error for seed {seed}: {e}\")\n            return None\n    \n    def analyze_difficulty_distribution(self, board_size: int, num_robots: int) -> Dict[str, any]:\n        \"\"\"Analyze difficulty distribution for a configuration.\"\"\"\n        lookup_table = self.difficulty_lookup.load_lookup_table(board_size, num_robots)\n        \n        if not lookup_table:\n            print(f\"No lookup table found for {board_size}x{board_size} with {num_robots} robots\")\n            return {}\n        \n        solve_lengths = list(lookup_table.values())\n        \n        analysis = {\n            \"total_puzzles\": len(solve_lengths),\n            \"min_solve_length\": min(solve_lengths),\n            \"max_solve_length\": max(solve_lengths),\n            \"mean_solve_length\": sum(solve_lengths) / len(solve_lengths),\n            \"median_solve_length\": sorted(solve_lengths)[len(solve_lengths) // 2],\n        }\n        \n        # Distribution by difficulty ranges\n        easy_count = sum(1 for length in solve_lengths if length <= 3)\n        medium_count = sum(1 for length in solve_lengths if 4 <= length <= 6)\n        hard_count = sum(1 for length in solve_lengths if 7 <= length <= 10)\n        expert_count = sum(1 for length in solve_lengths if length > 10)\n        \n        analysis[\"difficulty_distribution\"] = {\n            \"easy (1-3 moves)\": easy_count,\n            \"medium (4-6 moves)\": medium_count,\n            \"hard (7-10 moves)\": hard_count,\n            \"expert (11+ moves)\": expert_count\n        }\n        \n        return analysis\n    \n    def generate_standard_lookup_tables(self) -> None:\n        \"\"\"Generate standard lookup tables for common configurations.\"\"\"\n        configs = [\n            (4, 2, 5000),   # 4x4 board, 2 robots\n            (6, 2, 7500),   # 6x6 board, 2 robots\n            (8, 3, 10000),  # 8x8 board, 3 robots\n            (12, 4, 15000), # 12x12 board, 4 robots\n            (16, 4, 20000), # 16x16 board, 4 robots\n        ]\n        \n        for board_size, num_robots, num_samples in configs:\n            print(f\"\\n{'='*60}\")\n            lookup_table = self.generate_lookup_table(board_size, num_robots, num_samples)\n            analysis = self.analyze_difficulty_distribution(board_size, num_robots)\n            \n            print(f\"\\nAnalysis for {board_size}x{board_size} with {num_robots} robots:\")\n            for key, value in analysis.items():\n                if key == \"difficulty_distribution\":\n                    print(f\"  {key}:\")\n                    for subkey, subvalue in value.items():\n                        print(f\"    {subkey}: {subvalue}\")\n                else:\n                    print(f\"  {key}: {value}\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description=\"Generate difficulty lookup tables\")\n    parser.add_argument(\"--board-size\", type=int, help=\"Board size (square)\")\n    parser.add_argument(\"--num-robots\", type=int, help=\"Number of robots\")\n    parser.add_argument(\"--num-samples\", type=int, default=10000, help=\"Number of samples to generate\")\n    parser.add_argument(\"--lookup-dir\", type=str, default=\"difficulty_lookup\", help=\"Directory for lookup tables\")\n    parser.add_argument(\"--analyze\", action=\"store_true\", help=\"Analyze existing lookup table\")\n    parser.add_argument(\"--generate-standard\", action=\"store_true\", help=\"Generate standard lookup tables\")\n    \n    args = parser.parse_args()\n    \n    generator = DifficultyGenerator(args.lookup_dir)\n    \n    if args.generate_standard:\n        generator.generate_standard_lookup_tables()\n    elif args.analyze:\n        if not args.board_size or not args.num_robots:\n            print(\"Error: --board-size and --num-robots required for analysis\")\n            return\n        analysis = generator.analyze_difficulty_distribution(args.board_size, args.num_robots)\n        print(f\"Analysis for {args.board_size}x{args.board_size} with {args.num_robots} robots:\")\n        for key, value in analysis.items():\n            print(f\"  {key}: {value}\")\n    else:\n        if not args.board_size or not args.num_robots:\n            print(\"Error: --board-size and --num-robots required\")\n            return\n        generator.generate_lookup_table(args.board_size, args.num_robots, args.num_samples)\n\n\nif __name__ == \"__main__\":\n    main()