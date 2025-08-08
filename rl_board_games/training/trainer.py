from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..core.agent import Agent
from ..core.curriculum import Curriculum, ProgressiveCurriculum
from ..core.persistence import CheckpointManager
from .ricochet_robots_env import RicochetRobotsEnv


class WandbCallback(BaseCallback):
    """Callback to log metrics to Weights & Biases."""

    def __init__(self, curriculum: Optional[ProgressiveCurriculum] = None, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            # Log basic training metrics
            if hasattr(self.model, '_episode_num'):
                wandb.log({"episode": self.model._episode_num}, step=self.num_timesteps)
            
            # Log curriculum metrics if available
            if self.curriculum is not None:
                curriculum_metrics = self.curriculum.get_metrics()
                wandb.log(curriculum_metrics, step=self.num_timesteps)
        
        return True


class Trainer:
    """
    Main training orchestrator.
    """

    def __init__(
        self,
        agent: Agent,
        env: RicochetRobotsEnv,
        curriculum: Optional[ProgressiveCurriculum] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        use_wandb: bool = True,
        wandb_project: str = "rl-board-games",
        wandb_run_name: Optional[str] = None,
        curriculum_update_freq: int = 100,
    ):
        self.agent = agent
        self.env = env
        self.curriculum = curriculum
        self.checkpoint_manager = checkpoint_manager
        self.use_wandb = use_wandb
        self.curriculum_update_freq = curriculum_update_freq
        
        # Track curriculum episodes
        self._curriculum_episodes = []

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"train-{int(time.time())}",
                config=self._get_config(),
            )

    def _get_config(self) -> Dict[str, Any]:
        """Get configuration for logging."""
        config = {
            "agent_type": type(self.agent).__name__,
            "env_type": type(self.env).__name__,
            "max_episode_steps": self.env.max_episode_steps,
        }
        
        # Add SB3 model config if available
        if hasattr(self.agent, 'get_model'):
            model = self.agent.get_model()
            if hasattr(model, 'learning_rate'):
                config['learning_rate'] = model.learning_rate
            if hasattr(model, 'gamma'):
                config['gamma'] = model.gamma
                
        return config

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        eval_episodes: int = 10,
        log_freq: int = 1000,
    ) -> None:
        """
        Main training loop.
        """
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Wrap environment for SB3
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Set up callbacks
        callbacks = []
        if self.use_wandb:
            callbacks.append(WandbCallback(curriculum=self.curriculum, log_freq=log_freq))
            
        # Set environment on agent
        if hasattr(self.agent, 'set_env'):
            self.agent.set_env(vec_env)

        # Training loop with periodic evaluation
        timesteps_trained = 0
        
        while timesteps_trained < total_timesteps:
            # NOTE: CurriculumRicochetRobotsEnv already handles game switching
            # internally on every environment reset. Performing an additional
            # manual swap here replaced the underlying `game` instance
            # *without* resetting it, causing `Game not reset` runtime errors
            # during subsequent calls to `env.step()`. We therefore no longer
            # mutate the environment here; all curriculum progression is
            # delegated to the environment itself.
            
            # Determine training chunk size
            chunk_size = min(eval_freq, total_timesteps - timesteps_trained)
            
            # Train for this chunk
            current_level = self.curriculum.get_current_level().name if self.curriculum else "N/A"
            print(f"Training steps {timesteps_trained} to {timesteps_trained + chunk_size} [Curriculum: {current_level}]")
            
            self.agent.learn(
                total_timesteps=chunk_size,
                callback=callbacks if callbacks else None,
                reset_num_timesteps=False,
            )
            timesteps_trained += chunk_size
            
            # Evaluate
            if eval_freq > 0 and timesteps_trained % eval_freq == 0:
                eval_metrics = self._evaluate(eval_episodes, timesteps_trained)
                
                # Update curriculum based on evaluation results
                if self.curriculum is not None:
                    self._update_curriculum_progress(eval_metrics)
                
            # Save checkpoint
            if save_freq > 0 and timesteps_trained % save_freq == 0:
                self._save_checkpoint(timesteps_trained)
                
                # Save curriculum state
                if self.curriculum is not None:
                    self._save_curriculum_state(timesteps_trained)
                
        print("Training completed!")
        
        # Final evaluation and save
        final_metrics = self._evaluate(eval_episodes, timesteps_trained)
        if self.curriculum is not None:
            self._update_curriculum_progress(final_metrics)
        self._save_checkpoint(timesteps_trained, final=True)

    def _evaluate(self, num_episodes: int, timestep: int) -> Dict[str, float]:
        """Evaluate the agent."""
        current_level = self.curriculum.get_current_level().name if self.curriculum else "N/A"
        print(f"Evaluating for {num_episodes} episodes at timestep {timestep} [Curriculum: {current_level}]")
        
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            truncated = False
            
            while not done and not truncated and steps < self.env.max_episode_steps:
                # Use agent to predict action
                if hasattr(self.agent, 'predict'):
                    action, _ = self.agent.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()  # Fallback
                
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # Success is completing the episode (reaching goal) without truncation
            if done and not truncated:
                success_count += 1
                
        # Calculate metrics
        success_rate = success_count / num_episodes
        metrics = {
            "eval/mean_reward": sum(episode_rewards) / len(episode_rewards),
            "eval/mean_length": sum(episode_lengths) / len(episode_lengths),
            "eval/success_rate": success_rate,
            "eval/success_count": success_count,
            "timestep": timestep,
        }
        
        # Add curriculum-specific metrics
        if self.curriculum is not None:
            curriculum_metrics = self.curriculum.get_metrics()
            metrics.update(curriculum_metrics)
        
        print(f"Eval metrics: {metrics}")
        
        if self.use_wandb:
            wandb.log(metrics, step=timestep)
            
        return metrics

    def _save_checkpoint(self, timestep: int, final: bool = False) -> None:
        """Save model checkpoint."""
        if self.checkpoint_manager is None:
            return
            
        checkpoint_name = f"checkpoint_{timestep}" if not final else "final_model"
        print(f"Saving checkpoint: {checkpoint_name}")
        
        # Save agent
        self.agent.save(self.checkpoint_manager.root_dir / f"{checkpoint_name}.zip")
        
        # Save metadata
        metadata = {
            "timestep": timestep,
            "timestamp": time.time(),
            "final": final,
        }
        
        # Add curriculum state to metadata
        if self.curriculum is not None:
            metadata["curriculum_state"] = self.curriculum.get_state().to_dict()
            metadata["curriculum_level"] = self.curriculum.get_current_level().name
        
        self.checkpoint_manager.save_json(metadata, f"{checkpoint_name}_metadata")

    def _update_curriculum_progress(self, eval_metrics: Dict[str, float]) -> None:
        """Update curriculum progress based on evaluation results."""
        if self.curriculum is None:
            return
            
        # Extract success rate from evaluation metrics
        success_rate = eval_metrics.get("eval/success_rate", 0.0)
        
        # Update curriculum with success rate
        # We simulate multiple episodes worth of results
        num_episodes = eval_metrics.get("eval/success_count", 0)
        num_failures = eval_metrics.get("eval/num_episodes", 10) - num_episodes
        
        # Record results in curriculum
        for _ in range(int(num_episodes)):
            self.curriculum.record_episode_result(True)
        for _ in range(int(num_failures)):
            self.curriculum.record_episode_result(False)
    
    def _save_curriculum_state(self, timestep: int) -> None:
        """Save curriculum state to file."""
        if self.curriculum is None or self.checkpoint_manager is None:
            return
            
        state_file = self.checkpoint_manager.root_dir / f"curriculum_state_{timestep}.json"
        self.curriculum.save_state(state_file)
        print(f"Saved curriculum state to {state_file}")
    
    def load_curriculum_state(self, timestep: int) -> None:
        """Load curriculum state from file."""
        if self.curriculum is None or self.checkpoint_manager is None:
            return
            
        state_file = self.checkpoint_manager.root_dir / f"curriculum_state_{timestep}.json"
        if state_file.exists():
            self.curriculum.load_state(state_file)
            print(f"Loaded curriculum state from {state_file}")
        else:
            print(f"No curriculum state file found at {state_file}")
    
    def close(self) -> None:
        """Clean up resources."""
        if self.use_wandb:
            wandb.finish() 