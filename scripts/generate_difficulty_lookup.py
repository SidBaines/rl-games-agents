#!/usr/bin/env python3
"""
Generate difficulty lookup tables for curriculum learning.

This script creates pre-computed lookup tables that map seeds to solve lengths
for different board configurations, enabling efficient curriculum learning.

Usage:
    # Generate all standard lookup tables
    python scripts/generate_difficulty_lookup.py --generate-all
    
    # Generate specific configuration
    python scripts/generate_difficulty_lookup.py --board-size 8 --num-robots 3 --num-samples 10000
    
    # Analyze existing lookup table
    python scripts/generate_difficulty_lookup.py --analyze --board-size 8 --num-robots 3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.core.difficulty_generator import DifficultyGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate difficulty lookup tables for curriculum learning")
    parser.add_argument("--board-size", type=int, help="Board size (square)")
    parser.add_argument("--num-robots", type=int, help="Number of robots")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--lookup-dir", type=str, default="difficulty_lookup", help="Directory for lookup tables")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing lookup table")
    parser.add_argument("--generate-all", action="store_true", help="Generate all standard lookup tables")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed offset for generation")
    parser.add_argument("--max-walls-min", type=int, default=5, help="Minimum number of walls")
    parser.add_argument("--max-walls-max", type=int, default=25, help="Maximum number of walls")
    
    args = parser.parse_args()
    
    # Create lookup directory
    lookup_dir = Path(args.lookup_dir)
    lookup_dir.mkdir(exist_ok=True)
    
    generator = DifficultyGenerator(args.lookup_dir)
    
    if args.generate_all:
        print("Generating all standard lookup tables...")
        generator.generate_standard_lookup_tables()
        
    elif args.analyze:
        if not args.board_size or not args.num_robots:
            print("Error: --board-size and --num-robots required for analysis")
            sys.exit(1)
            
        print(f"Analyzing lookup table for {args.board_size}x{args.board_size} with {args.num_robots} robots...")
        analysis = generator.analyze_difficulty_distribution(args.board_size, args.num_robots)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            sys.exit(1)
        
        print(f"\\nAnalysis Results:")
        print(f"  Total puzzles: {analysis['total_puzzles']}")
        print(f"  Solve length range: {analysis['min_solve_length']} - {analysis['max_solve_length']}")
        print(f"  Mean solve length: {analysis['mean_solve_length']:.2f}")
        print(f"  Median solve length: {analysis['median_solve_length']}")
        print(f"\\nDifficulty Distribution:")
        for category, count in analysis['difficulty_distribution'].items():
            percentage = (count / analysis['total_puzzles']) * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")
            
    else:
        if not args.board_size or not args.num_robots:
            print("Error: --board-size and --num-robots required for generation")
            print("Use --generate-all to generate all standard configurations")
            sys.exit(1)
            
        print(f"Generating lookup table for {args.board_size}x{args.board_size} with {args.num_robots} robots...")
        
        lookup_table = generator.generate_lookup_table(
            board_size=args.board_size,
            num_robots=args.num_robots,
            num_samples=args.num_samples,
            max_walls_range=(args.max_walls_min, args.max_walls_max),
            seed_offset=args.seed_offset
        )
        
        print(f"\\nGeneration completed!")
        print(f"Lookup table saved with {len(lookup_table)} entries")
        
        # Show quick analysis
        analysis = generator.analyze_difficulty_distribution(args.board_size, args.num_robots)
        if analysis and "error" not in analysis:
            print(f"\\nQuick Analysis:")
            print(f"  Solve length range: {analysis['min_solve_length']} - {analysis['max_solve_length']}")
            print(f"  Mean solve length: {analysis['mean_solve_length']:.2f}")

    print(f"\\nLookup tables stored in: {lookup_dir.absolute()}")


if __name__ == "__main__":
    main()