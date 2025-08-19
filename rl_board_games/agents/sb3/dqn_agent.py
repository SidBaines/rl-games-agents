from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from stable_baselines3 import DQN

from ...core.encoder import Encoder
from .sb3_agent import SB3Agent


class DQNAgent(SB3Agent):
    """DQN agent wrapper."""

    def __init__(
        self,
        env,
        encoder: Encoder,
        policy: str = "MlpPolicy",
        policy_kwargs: Dict[str, Any] | None = None,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        verbose: int = 1,
        **kwargs
    ):
        model = DQN(
            policy,
            env,
            policy_kwargs=policy_kwargs or {},
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            verbose=verbose,
            **kwargs
        )
        super().__init__(model, encoder)

    @classmethod
    def load_from_checkpoint(cls, path: str | Path, env, encoder: Encoder) -> "DQNAgent":
        """Load a pre-trained DQN model."""
        model = DQN.load(str(path), env=env)
        return cls.__new__(cls)._init_from_model(model, encoder)

    def _init_from_model(self, model: DQN, encoder: Encoder) -> "DQNAgent":
        """Initialize from an existing model (used by load_from_checkpoint)."""
        self.model = model
        self.encoder = encoder
        return self 