#!/usr/bin/env python3
"""
Quick demo of the training pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder
from rl_board_games.games.ricochet_robots.board import Board
from rl_board_games.agents.sb3 import DQNAgent
from rl_board_games.training.ricochet_robots_env import RicochetRobotsEnv
from rl_board_games.training.trainer import Trainer
from rl_board_games.core.persistence import CheckpointManager


def main():
    print("Creating Ricochet Robots training demo...")
    
    # Create components
    board = Board.empty(8)  # Small board for quick demo
    game = RicochetRobotsGame(board=board, num_robots=4)
    encoder = FlatArrayEncoder()
    env = RicochetRobotsEnv(game, encoder, max_episode_steps=20)
    
    # Create DQN agent with small buffer for demo
    agent = DQNAgent(
        env=env,
        encoder=encoder,
        learning_rate=1e-3,
        buffer_size=1000,
        learning_starts=100,
        batch_size=16,
    )
    
    # Setup checkpointing
    checkpoint_manager = CheckpointManager("./demo_checkpoints/")
    
    # Create trainer without wandb
    trainer = Trainer(
        agent=agent,
        env=env,
        checkpoint_manager=checkpoint_manager,
        use_wandb=False,
    )
    
    print("Starting short training run...")
    try:
        trainer.train(
            total_timesteps=1000,  # Very short for demo
            eval_freq=500,
            save_freq=1000,
            eval_episodes=3,
        )
    finally:
        trainer.close()
    
    print("Demo completed!")


if __name__ == "__main__":
    main() 