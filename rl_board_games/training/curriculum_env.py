from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np
from gymnasium import spaces

from ..core.curriculum import ProgressiveCurriculum
from ..core.encoder import Encoder
from ..games.ricochet_robots import RicochetRobotsGame
from .ricochet_robots_env import RicochetRobotsEnv


class CurriculumRicochetRobotsEnv(RicochetRobotsEnv):
    """
    Curriculum-aware Ricochet Robots environment.
    
    This environment integrates with a curriculum to dynamically adjust
    the game difficulty based on agent performance.
    """
    
    def __init__(
        self,
        curriculum: ProgressiveCurriculum,
        encoder: Encoder,
        max_episode_steps: int = 100,
        curriculum_update_freq: int = 1,
    ):
        self.curriculum = curriculum
        self.curriculum_update_freq = curriculum_update_freq
        self.episode_count = 0
        self.current_game = None
        
        # Initialize with first game from curriculum
        self._update_game_from_curriculum()
        
        # Initialize parent with current game
        super().__init__(self.current_game, encoder, max_episode_steps)
    
    def _update_game_from_curriculum(self) -> None:
        """Update the current game from curriculum."""
        curriculum_iter = iter(self.curriculum)
        self.current_game = next(curriculum_iter)

        # Keep track of the seed that produced this game so that we can
        # reliably reset to the *identical* starting position that the
        # curriculum evaluated difficulty on.
        self.current_seed = getattr(self.current_game, "initial_seed", None)
        
        # Update game references
        self.game = self.current_game
        self.ricochet_game = self.current_game
        
        # Update action space if needed (in case number of robots changed)
        self._setup_action_space()
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and potentially update curriculum."""
        # Update curriculum game periodically
        if self.episode_count % self.curriculum_update_freq == 0:
            self._update_game_from_curriculum()
        
        self.episode_count += 1

        # If no explicit seed provided, fall back to the deterministic seed
        # that generated this curriculum game.  This guarantees that every
        # reset reproduces a state that is solvable within the curriculum's
        # prescribed number of moves.
        seed_to_use = seed if seed is not None else self.current_seed

        # Reset with current game
        return super().reset(seed=seed_to_use, options=options)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and track results for curriculum."""
        obs, reward, done, truncated, info = super().step(action)
        
        # Add curriculum info
        current_level = self.curriculum.get_current_level()
        info.update({
            "curriculum_level": current_level.name,
            "curriculum_level_index": self.curriculum.state.current_level,
            "curriculum_success_rate": self.curriculum.state.success_rate,
            "curriculum_episodes": self.curriculum.state.total_episodes,
        })
        
        # Record episode result in curriculum if episode finished
        if done or truncated:
            success = done and not truncated  # Success if done without truncation
            self.curriculum.record_episode_result(success)
            
            # Add curriculum progression info
            info.update({
                "curriculum_episode_success": success,
                "curriculum_updated": self.curriculum.state.level_episodes == 1,  # Just advanced
            })
        
        return obs, reward, done, truncated, info
    
    def get_curriculum_metrics(self) -> Dict[str, Any]:
        """Get curriculum metrics for logging."""
        return self.curriculum.get_metrics()
    
    def save_curriculum_state(self, path: str) -> None:
        """Save curriculum state."""
        self.curriculum.save_state(path)
    
    def load_curriculum_state(self, path: str) -> None:
        """Load curriculum state."""
        self.curriculum.load_state(path)
    
    def get_current_level_info(self) -> Dict[str, Any]:
        """Get information about current curriculum level."""
        level = self.curriculum.get_current_level()
        return {
            "name": level.name,
            "board_size": level.board_size,
            "num_robots": level.num_robots,
            "difficulty_range": (level.min_solve_length, level.max_solve_length),
            "success_threshold": level.success_threshold,
        }