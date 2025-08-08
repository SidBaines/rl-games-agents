from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from ...core.agent import Agent
from ...core.game import GameState
from ...core.encoder import Encoder


class SB3Agent(Agent):
    """
    Wrapper around any Stable-Baselines3 model to implement our Agent interface.
    """

    def __init__(self, model: BaseAlgorithm, encoder: Encoder):
        self.model = model
        self.encoder = encoder

    def act(self, state: GameState) -> Any:
        """Convert state to observation and predict action."""
        obs = self.encoder.encode(state)
        # Ensure obs is in the right shape for SB3 (add batch dimension if needed)
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            obs = obs.reshape(1, -1)
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:  # CHW format
            obs = obs.reshape(1, *obs.shape)
        
        action, _ = self.model.predict(obs, deterministic=True)
        # SB3 returns action in batch format, extract single action
        if isinstance(action, np.ndarray) and action.ndim > 0:
            return action.item() if action.size == 1 else action[0]
        return action

    def learn(self, total_timesteps: int = 10000, **kwargs) -> None:
        """Train the SB3 model."""
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save the SB3 model."""
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        """Load the SB3 model."""
        # Note: SB3 models need to be loaded differently - this is a simplified approach
        # In practice, you'd need to know the model class to load properly
        self.model = self.model.load(str(path))

    def set_env(self, env) -> None:
        """Set the environment for the SB3 model."""
        self.model.set_env(env)

    def get_model(self) -> BaseAlgorithm:
        """Access underlying SB3 model."""
        return self.model 