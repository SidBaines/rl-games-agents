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

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder, PlanarEncoder, WallAwarePlanarEncoder
from rl_board_games.games.ricochet_robots.board import Board
from rl_board_games.agents.sb3 import DQNAgent, PPOAgent
from rl_board_games.training.curriculum_env import CurriculumRicochetRobotsEnv
from rl_board_games.training.trainer import Trainer
from rl_board_games.core.persistence import CheckpointManager
from rl_board_games.core.curriculum import DifficultyLookup, CurriculumLevel
from rl_board_games.curricula.ricochet_robots_curriculum import RicochetRobotsCurriculum


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
            episodes_per_evaluation=level_config["episodes_per_evaluation"]
        )
        levels.append(level)
    return levels


def create_curriculum(config: dict) -> RicochetRobotsCurriculum:
    """Create curriculum instance from config."""
    curriculum_config = config["curriculum"]
    
    # Create difficulty lookup
    lookup_dir = curriculum_config.get("difficulty_lookup_dir", "./difficulty_lookup")
    difficulty_lookup = DifficultyLookup(lookup_dir)
    
    # Create curriculum levels
    levels = create_curriculum_levels(config)
    
    # Create curriculum
    curriculum = RicochetRobotsCurriculum(
        levels=levels,
        difficulty_lookup=difficulty_lookup,
        evaluation_episodes=curriculum_config.get("evaluation_episodes", 50),
        max_fallback_attempts=curriculum_config.get("max_fallback_attempts", 5)
    )
    
    return curriculum


def create_encoder(config: dict):
    """Create encoder instance from config."""
    encoder_config = config["encoder"]
    
    if encoder_config["type"] == "flat_array":
        return FlatArrayEncoder()
    elif encoder_config["type"] == "planar":
        # return PlanarEncoder()
        return WallAwarePlanarEncoder()
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
    print(f"Created curriculum environment")

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
        
        print(f"Starting curriculum training...")
        print(f"Total timesteps: {training_config['total_timesteps']}")
        print(f"Initial level: {curriculum.get_current_level().name}")
        
        trainer.train(
            total_timesteps=training_config["total_timesteps"],
            eval_freq=training_config["eval_freq"],
            save_freq=training_config["save_freq"],
            eval_episodes=training_config["eval_episodes"],
            log_freq=training_config["log_freq"],
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