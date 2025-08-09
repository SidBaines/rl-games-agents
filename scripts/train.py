#!/usr/bin/env python3
"""
Main training script for RL board games.

Usage:
    python scripts/train.py configs/ricochet_robots/dqn_small.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder, PlanarEncoder, WallAwarePlanarEncoder, RGBArrayEncoder
from rl_board_games.games.ricochet_robots.board import Board
from rl_board_games.agents.sb3 import DQNAgent, PPOAgent
from rl_board_games.training.ricochet_robots_env import RicochetRobotsEnv
from rl_board_games.training.trainer import Trainer
from rl_board_games.core.persistence import CheckpointManager


def create_game(config: dict):
    """Create game instance from config."""
    game_config = config["game"]
    
    if game_config["type"] == "ricochet_robots":
        board = Board.empty(game_config["board_size"])
        return RicochetRobotsGame(
            board=board,
            num_robots=game_config["num_robots"],
            reward_per_move=game_config["reward_per_move"],
            reward_goal=game_config["reward_goal"],
        )
    else:
        raise ValueError(f"Unknown game type: {game_config['type']}")


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


def create_environment(game, encoder, config: dict):
    """Create environment instance from config."""
    env_config = config["environment"]
    return RicochetRobotsEnv(
        game=game,
        encoder=encoder,
        max_episode_steps=env_config["max_episode_steps"],
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
            # verbose=agent_config.get("verbose", 0),
            verbose=0,
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
    parser = argparse.ArgumentParser(description="Train RL agents on board games")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
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

    # Create components
    game = create_game(config)
    encoder = create_encoder(config)
    env = create_environment(game, encoder, config)
    agent = create_agent(env, encoder, config)

    # Setup checkpointing
    checkpoint_dir = Path(config["checkpoints"]["save_dir"])
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    # Create trainer
    logging_config = config["logging"]
    use_wandb = logging_config["use_wandb"] and not args.no_wandb
    
    trainer = Trainer(
        agent=agent,
        env=env,
        checkpoint_manager=checkpoint_manager,
        use_wandb=use_wandb,
        wandb_project=logging_config["wandb_project"],
        wandb_run_name=logging_config["wandb_run_name"],
    )

    try:
        # Start training
        training_config = config["training"]
        trainer.train(
            total_timesteps=training_config["total_timesteps"],
            eval_freq=training_config["eval_freq"],
            save_freq=training_config["save_freq"],
            eval_episodes=training_config["eval_episodes"],
            log_freq=training_config.get("log_freq", 1000),
            rollout_log_freq=training_config.get("rollout_log_freq", 0),
            rollout_max_steps=training_config.get("rollout_max_steps", 50),
            rollout_fps=training_config.get("rollout_fps", 2),
            profile=bool(args.profile or training_config.get("profile", False)),
            profile_output=args.profile_output or training_config.get("profile_output"),
        )
    finally:
        trainer.close()

    print("Training completed successfully!")


if __name__ == "__main__":
    main() 