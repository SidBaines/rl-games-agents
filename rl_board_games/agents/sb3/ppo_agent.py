from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import PPO

from ...core.encoder import Encoder
from .sb3_agent import SB3Agent


class PPOAgent(SB3Agent):
    """PPO agent wrapper."""

    def __init__(
        self,
        env,
        encoder: Encoder,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
        **kwargs
    ):
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            **kwargs
        )
        super().__init__(model, encoder)

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, env, encoder: Encoder) -> "PPOAgent":
        """Load a pre-trained PPO model."""
        model = PPO.load(str(path), env=env)
        return cls.__new__(cls)._init_from_model(model, encoder)

    def _init_from_model(self, model: PPO, encoder: Encoder) -> "PPOAgent":
        """Initialize from an existing model (used by load_from_checkpoint)."""
        self.model = model
        self.encoder = encoder
        return self 