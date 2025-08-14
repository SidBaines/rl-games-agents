#!/usr/bin/env python3
"""
Curriculum training script for RL board games.

This script enables curriculum learning with progressive difficulty,
allowing agents to learn from easy to hard scenarios.

Usage:
    python scripts/train_curriculum.py configs/ricochet_robots/dqn_curriculum.yaml
    python scripts/train_curriculum.py configs/ricochet_robots/dqn_curriculum.yaml --no-wandb
"""

import argparse
import sys
from pathlib import Path
import datetime

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.games.ricochet_robots import FlatArrayEncoder, WallAwarePlanarEncoder, RGBArrayEncoder
from rl_board_games.agents.sb3 import DQNAgent, PPOAgent
from rl_board_games.training.curriculum_env import CurriculumRicochetRobotsEnv
from rl_board_games.training.trainer import Trainer
from rl_board_games.core.persistence import CheckpointManager
from rl_board_games.core.curriculum import DifficultyLookup, CurriculumLevel
from rl_board_games.curricula.ricochet_robots_curriculum import RicochetRobotsCurriculum
from rl_board_games.curricula.astar_plan_curriculum import AStarPlanCurriculum, PlanCurriculumLevel
from rl_board_games.core.plan_cache import PlanDifficultyCache


def create_curriculum_levels(config: dict) -> list[CurriculumLevel]:
    """Create curriculum levels from configuration."""
    levels = []
    for level_config in config["curriculum"]["levels"]:
        level = CurriculumLevel(
            name=level_config["name"],
            min_solve_length=level_config["min_solve_length"],
            max_solve_length=level_config["max_solve_length"],
            success_threshold=level_config["success_threshold"],
            board_size=level_config["board_size"],
            num_robots=level_config["num_robots"],
            max_walls=level_config["max_walls"],
            episodes_per_evaluation=level_config["episodes_per_evaluation"],
            board_size_min=level_config.get("board_size_min"),
            board_size_max=level_config.get("board_size_max"),
            max_episode_steps=level_config.get("max_episode_steps"),
        )
        levels.append(level)
    return levels


def create_plan_curriculum_levels(config: dict) -> list[PlanCurriculumLevel]:
    """Create A* plan curriculum levels from configuration, if provided."""
    levels: list[PlanCurriculumLevel] = []
    for level_config in config["curriculum"].get("levels", []):
        level = PlanCurriculumLevel(
            name=level_config["name"],
            min_solve_length=level_config["min_solve_length"],
            max_solve_length=level_config["max_solve_length"],
            success_threshold=level_config["success_threshold"],
            board_size=level_config["board_size"],
            num_robots=level_config["num_robots"],
            max_walls=level_config["max_walls"],
            episodes_per_evaluation=level_config["episodes_per_evaluation"],
            board_size_min=level_config.get("board_size_min"),
            board_size_max=level_config.get("board_size_max"),
            max_total_moves=level_config.get("max_total_moves", 1),
            max_robots_moved=level_config.get("max_robots_moved", 1),
            min_total_moves=level_config.get("min_total_moves", 0),
            min_robots_moved=level_config.get("min_robots_moved", 0),
            max_episode_steps=level_config.get("max_episode_steps"),
        )
        levels.append(level)
    return levels


def create_curriculum(config: dict) -> RicochetRobotsCurriculum:
    """Create curriculum instance from config."""
    curriculum_config = config["curriculum"]
    
    # Create difficulty lookup
    lookup_dir = curriculum_config.get("difficulty_lookup_dir", "./difficulty_lookup")
    difficulty_lookup = DifficultyLookup(lookup_dir)
    
    # Create curriculum levels (for base curriculum), and plan levels for A* if needed
    levels = create_curriculum_levels(config)
    
    # Create curriculum by type
    cur_type = curriculum_config.get("type", "ricochet_robots")
    if cur_type == "ricochet_robots":
        curriculum = RicochetRobotsCurriculum(
            levels=levels,
            difficulty_lookup=difficulty_lookup,
            evaluation_episodes=curriculum_config.get("evaluation_episodes", 50),
            max_fallback_attempts=curriculum_config.get("max_fallback_attempts", 5),
        )
    elif cur_type == "astar_plan":
        plan_levels = create_plan_curriculum_levels(config)
        curriculum = AStarPlanCurriculum(
            levels=plan_levels if plan_levels else None,
            evaluation_episodes=curriculum_config.get("evaluation_episodes", 50),
        )
    else:
        raise ValueError(f"Unknown curriculum type: {cur_type}")
    
    return curriculum


def _assert_sufficient_plan_cache(config: dict, curriculum: RicochetRobotsCurriculum) -> None:
    """Check the plan cache has at least N matching seeds per level; raise if not."""
    curriculum_cfg = config["curriculum"]
    if curriculum_cfg.get("type") != "astar_plan":
        return

    min_per_level = int(curriculum_cfg.get("min_cached_seeds_per_level", 5))
    plan_cache_dir = curriculum_cfg.get("plan_cache_dir", "plan_lookup")
    cache = PlanDifficultyCache(plan_cache_dir)

    # Use A* plan levels for constraints
    if hasattr(curriculum, "levels") and curriculum.levels and isinstance(curriculum.levels[0], PlanCurriculumLevel):
        plan_levels: list[PlanCurriculumLevel] = curriculum.levels  # type: ignore[assignment]
    else:
        # Build from config as fallback
        plan_levels = create_plan_curriculum_levels(config)

    insufficient = []
    for idx, level in enumerate(plan_levels):
        # Count across all board sizes allowed by the level
        def count_for_size(board_size: int) -> int:
            matches = cache.get_matching_seeds(
                board_size=board_size,
                num_robots=int(level.num_robots),
                predicate=lambda feats: (
                    feats.get("total_moves", 10**9) <= int(min(level.max_total_moves, level.max_solve_length))
                    and feats.get("total_moves", -1) >= int(max(level.min_total_moves, level.min_solve_length))
                    and feats.get("robots_moved", 10**9) <= int(level.max_robots_moved)
                    and feats.get("robots_moved", -1) >= int(level.min_robots_moved)
                ),
            )
            return len(matches)

        sizes = [int(level.board_size)]
        if level.board_size_min is not None and level.board_size_max is not None:
            sizes = list(range(int(level.board_size_min), int(level.board_size_max) + 1))

        total_matches = sum(count_for_size(s) for s in sizes)
        if total_matches < min_per_level:
            insufficient.append((idx, level.name, total_matches))

    if insufficient:
        lines = [
            "Insufficient plan cache seeds for one or more levels:",
            *[f"  Level {idx} {name}: {count} < required {min_per_level}" for idx, name, count in insufficient],
            "Run scripts/generate_plan_cache.py <config> to pre-populate the cache.",
        ]
        raise RuntimeError("\n".join(lines))


def create_encoder(config: dict):
    """Create encoder instance from config."""
    encoder_config = config["encoder"]
    
    if encoder_config["type"] == "flat_array":
        return FlatArrayEncoder()
    elif encoder_config["type"] == "planar":
        # return PlanarEncoder()
        return WallAwarePlanarEncoder()
    elif encoder_config["type"] == "rgb":
        return RGBArrayEncoder(scale=encoder_config.get("scale", 20))
    else:
        raise ValueError(f"Unknown encoder type: {encoder_config['type']}")


def create_environment(curriculum: RicochetRobotsCurriculum, encoder, config: dict):
    """Create curriculum environment instance from config."""
    env_config = config["environment"]
    return CurriculumRicochetRobotsEnv(
        curriculum=curriculum,
        encoder=encoder,
        max_episode_steps=env_config["max_episode_steps"],
        curriculum_update_freq=env_config.get("curriculum_update_freq", 1)
    )


def create_agent(env, encoder, config: dict):
    """Create agent instance from config."""
    agent_config = config["agent"]
    
    if agent_config["type"] == "dqn":
        return DQNAgent(
            env=env,
            encoder=encoder,
            learning_rate=agent_config["learning_rate"],
            buffer_size=agent_config["buffer_size"],
            learning_starts=agent_config["learning_starts"],
            batch_size=agent_config["batch_size"],
            gamma=agent_config["gamma"],
            exploration_fraction=agent_config["exploration_fraction"],
            exploration_initial_eps=agent_config["exploration_initial_eps"],
            exploration_final_eps=agent_config["exploration_final_eps"],
            verbose=agent_config.get("verbose", 0),
        )
    elif agent_config["type"] == "ppo":
        return PPOAgent(
            env=env,
            encoder=encoder,
            learning_rate=agent_config.get("learning_rate", 3e-4),
            n_steps=agent_config.get("n_steps", 2048),
            batch_size=agent_config.get("batch_size", 64),
            gamma=agent_config.get("gamma", 0.99),
            verbose=agent_config.get("verbose", 0),
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")


def main():
    parser = argparse.ArgumentParser(description="Train RL agents with curriculum learning")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint (timestep)")
    parser.add_argument("--generate-lookup", action="store_true", help="Generate difficulty lookup tables first")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile during training")
    parser.add_argument("--profile-output", type=str, help="Path to write .prof profile stats")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    print(f"Loaded configuration from {config_path}")
    print(f"Config: {config}")

    # Generate lookup tables if requested
    if args.generate_lookup:
        print("Generating difficulty lookup tables...")
        from rl_board_games.core.difficulty_generator import DifficultyGenerator
        
        lookup_dir = config["curriculum"].get("difficulty_lookup_dir", "./difficulty_lookup")
        generator = DifficultyGenerator(lookup_dir)
        generator.generate_standard_lookup_tables()
        print("Lookup tables generated successfully!")

    # Create curriculum
    curriculum = create_curriculum(config)
    print(f"Created curriculum with {len(curriculum.levels)} levels")
    
    # Print curriculum levels
    for i, level in enumerate(curriculum.levels):
        print(f"  Level {i}: {level.name} - {level.board_size}x{level.board_size}, "
              f"{level.num_robots} robots, difficulty {level.min_solve_length}-{level.max_solve_length}")

    # Create encoder
    encoder = create_encoder(config)

    # Create environment
    env = create_environment(curriculum, encoder, config)
    print("Created curriculum environment")

    # Verify plan cache sufficiency before training (for astar_plan)
    try:
        _assert_sufficient_plan_cache(config, curriculum)
    except RuntimeError as e:
        print(str(e))
        sys.exit(2)

    # Create agent
    agent = create_agent(env, encoder, config)
    print(f"Created agent: {type(agent).__name__}")

    # Setup checkpointing
    checkpoint_dir = Path(config["checkpoints"]["save_dir"])
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create trainer
    logging_config = config["logging"]
    use_wandb = logging_config["use_wandb"] and not args.no_wandb
    
    trainer = Trainer(
        agent=agent,
        env=env,
        curriculum=curriculum,
        checkpoint_manager=checkpoint_manager,
        use_wandb=use_wandb,
        wandb_project=logging_config["wandb_project"],
        wandb_run_name=f"{logging_config['wandb_run_name']}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        curriculum_update_freq=config["training"].get("curriculum_update_freq", 100)
    )

    # Resume from checkpoint if requested
    if args.resume:
        try:
            timestep = int(args.resume)
            trainer.load_curriculum_state(timestep)
            print(f"Resumed curriculum from timestep {timestep}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Could not resume from checkpoint: {e}")

    try:
        # Start training
        training_config = config["training"]
        
        print("Starting curriculum training...")
        print(f"Total timesteps: {training_config['total_timesteps']}")
        print(f"Initial level: {curriculum.get_current_level().name}")
        
        trainer.train(
            total_timesteps=training_config["total_timesteps"],
            eval_freq=training_config["eval_freq"],
            save_freq=training_config["save_freq"],
            eval_episodes=training_config["eval_episodes"],
            log_freq=training_config["log_freq"],
            rollout_log_freq=training_config.get("rollout_log_freq", 0),
            rollout_max_steps=training_config.get("rollout_max_steps", 50),
            rollout_fps=training_config.get("rollout_fps", 2),
            profile=bool(args.profile or training_config.get("profile", False)),
            profile_output=args.profile_output or training_config.get("profile_output"),
        )
    finally:
        trainer.close()

    print("Curriculum training completed successfully!")
    
    # Print final curriculum status
    final_level = curriculum.get_current_level()
    print(f"Final curriculum level: {final_level.name}")
    print(f"Final success rate: {curriculum.state.success_rate:.3f}")
    print(f"Total episodes: {curriculum.state.total_episodes}")


if __name__ == "__main__":
    main()