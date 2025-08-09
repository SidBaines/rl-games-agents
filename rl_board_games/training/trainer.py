from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import cProfile
from time import perf_counter

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
        rollout_log_freq: int = 0,
        rollout_max_steps: int = 50,
        rollout_fps: int = 2,
        profile: bool = False,
        profile_output: Optional[str] = None,
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

        # Optional CPU profiler
        profiler = None
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

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
            
            t_train_start = perf_counter()
            self.agent.learn(
                total_timesteps=chunk_size,
                callback=callbacks if callbacks else None,
                reset_num_timesteps=False,
            )
            train_chunk_s = perf_counter() - t_train_start
            timesteps_trained += chunk_size
            # Log timing for training chunk
            train_timing = {
                "time/train_chunk_s": train_chunk_s,
                "time/train_per_step_ms": (train_chunk_s / max(chunk_size, 1)) * 1000.0,
            }
            print(f"[timing] train_chunk_s={train_chunk_s:.3f}s | per_step_ms={(train_chunk_s/max(chunk_size,1))*1000.0:.3f}")
            if self.use_wandb:
                wandb.log(train_timing, step=timesteps_trained)
            
            # Optional: log a rollout video/images to wandb
            if (
                self.use_wandb
                and rollout_log_freq > 0
                and timesteps_trained % rollout_log_freq == 0
            ):
                t_rollout_start = perf_counter()
                self._log_rollout(timestep=timesteps_trained, max_steps=rollout_max_steps, fps=rollout_fps)
                rollout_s = perf_counter() - t_rollout_start
                # Also log how long the rollout logging took
                if self.use_wandb:
                    wandb.log({"time/rollout_log_s": rollout_s}, step=timesteps_trained)
                print(f"[timing] rollout_log_s={rollout_s:.3f}s")
            
            # Evaluate
            if eval_freq > 0 and timesteps_trained % eval_freq == 0:
                t_eval_start = perf_counter()
                eval_metrics = self._evaluate(eval_episodes, timesteps_trained)
                eval_s = perf_counter() - t_eval_start
                # Log wallclock for evaluation wrapper
                if self.use_wandb:
                    wandb.log({"time/eval_total_wall_s": eval_s}, step=timesteps_trained)
                print(f"[timing] eval_total_wall_s={eval_s:.3f}s")
                
                # Update curriculum based on evaluation results
                if self.curriculum is not None:
                    t_cur_update = perf_counter()
                    self._update_curriculum_progress(eval_metrics)
                    cur_update_s = perf_counter() - t_cur_update
                    if self.use_wandb:
                        wandb.log({"time/curriculum_update_s": cur_update_s}, step=timesteps_trained)
                    print(f"[timing] curriculum_update_s={cur_update_s:.3f}s")
                
            # Save checkpoint
            if save_freq > 0 and timesteps_trained % save_freq == 0:
                t_ckpt = perf_counter()
                self._save_checkpoint(timesteps_trained)
                ckpt_s = perf_counter() - t_ckpt
                if self.use_wandb:
                    wandb.log({"time/checkpoint_save_s": ckpt_s}, step=timesteps_trained)
                print(f"[timing] checkpoint_save_s={ckpt_s:.3f}s")
                
                # Save curriculum state
                if self.curriculum is not None:
                    t_cur_save = perf_counter()
                    self._save_curriculum_state(timesteps_trained)
                    cur_save_s = perf_counter() - t_cur_save
                    if self.use_wandb:
                        wandb.log({"time/curriculum_state_save_s": cur_save_s}, step=timesteps_trained)
                    print(f"[timing] curriculum_state_save_s={cur_save_s:.3f}s")
                
        print("Training completed!")
        
        # Final evaluation and save
        final_metrics = self._evaluate(eval_episodes, timesteps_trained)
        if self.curriculum is not None:
            self._update_curriculum_progress(final_metrics)
        self._save_checkpoint(timesteps_trained, final=True)

        # Dump profiler stats if enabled
        if profiler is not None:
            profiler.disable()
            out_path = Path(profile_output) if profile_output else Path("profiles") / f"train_profile_{int(time.time())}.prof"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.dump_stats(str(out_path))
            print(f"Saved cProfile stats to {out_path}")

    def _evaluate(self, num_episodes: int, timestep: int) -> Dict[str, float]:
        """Evaluate the agent."""
        current_level = self.curriculum.get_current_level().name if self.curriculum else "N/A"
        print(f"Evaluating for {num_episodes} episodes at timestep {timestep} [Curriculum: {current_level}]")
        
        t_eval_start = perf_counter()
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            t_ep_start = perf_counter()
            obs, info = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            truncated = False

            while not done and not truncated and steps < self.env.max_episode_steps:
                # Use underlying SB3 model if available
                action = None
                model = self.agent.get_model() if hasattr(self.agent, 'get_model') else None
                if model is not None and hasattr(model, 'predict'):
                    try:
                        pred = model.predict(obs, deterministic=True)
                        if isinstance(pred, tuple):
                            action = pred[0]
                        else:
                            action = pred
                    except TypeError:
                        action = None
                
                if action is None and hasattr(self.agent, 'predict'):
                    try:
                        pred = self.agent.predict(obs, deterministic=True)
                        action = pred[0] if isinstance(pred, tuple) else pred
                    except Exception:
                        action = None
                
                # Coerce action to a plain int if it's a NumPy array or scalar-like
                if action is not None:
                    try:
                        if isinstance(action, np.ndarray):
                            action = int(action.item()) if action.size == 1 else int(action.flat[0])
                        else:
                            action = int(action)
                    except Exception:
                        action = None
                
                if action is None:
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
            "eval/num_episodes": num_episodes,
            "timestep": timestep,
        }
        
        # Add curriculum-specific metrics
        if self.curriculum is not None:
            curriculum_metrics = self.curriculum.get_metrics()
            metrics.update(curriculum_metrics)
        
        # Timing metrics for evaluation
        eval_elapsed = perf_counter() - t_eval_start
        total_steps = int(sum(episode_lengths)) if episode_lengths else 0
        metrics.update({
            "time/eval_total_s": eval_elapsed,
            "time/eval_avg_episode_ms": (eval_elapsed / max(num_episodes, 1)) * 1000.0,
            "time/eval_avg_step_ms": (eval_elapsed / max(total_steps, 1)) * 1000.0,
            "eval/steps_total": total_steps,
        })
        
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

    def _log_rollout(self, timestep: int, max_steps: int = 50, fps: int = 2) -> None:
        """Run a single-episode rollout and log frames/video to Weights & Biases."""
        try:
            t0 = perf_counter()
            frames: list[np.ndarray] = []
            obs, _ = self.env.reset()
            # Capture initial state
            t_render = 0.0
            t_r0 = perf_counter()
            first_frame = self.env.render(mode="rgb_array")
            t_render += perf_counter() - t_r0
            if first_frame is None:
                return
            frames.append(first_frame)

            total_reward = 0.0
            done = False
            truncated = False
            steps = 0

            # Prefer SB3 model if available
            model = self.agent.get_model() if hasattr(self.agent, 'get_model') else None

            while not done and not truncated and steps < max_steps:
                t_step0 = perf_counter()
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = self.env.action_space.sample()
                obs, reward, done, truncated, _ = self.env.step(action)
                step_elapsed = perf_counter() - t_step0
                total_reward += float(reward)
                steps += 1
                t_r = perf_counter()
                frame = self.env.render(mode="rgb_array")
                t_render += perf_counter() - t_r
                if frame is not None:
                    frames.append(frame)
                else:
                    break

            # Stack and log as video to keep storage reasonable
            if frames:
                video = np.stack(frames, axis=0)  # (T, H, W, C)
                # wandb.Video expects (T, C, H, W)
                video_chw = np.transpose(video, (0, 3, 1, 2))
                elapsed = perf_counter() - t0
                rollout_data = {
                    "rollout/video": wandb.Video(video_chw, fps=fps, format="gif"),
                    "rollout/length": steps,
                    "rollout/return": total_reward,
                    "rollout/success": bool(done and not truncated),
                    "rollout/time_total_s": elapsed,
                    "rollout/time_render_s": t_render,
                    "rollout/time_avg_frame_render_ms": (t_render / max(len(frames), 1)) * 1000.0,
                }
                wandb.log(rollout_data, step=timestep)
        except Exception as e:
            # Never break training due to logging errors
            print(f"[warn] Failed to log rollout at step {timestep}: {e}") 