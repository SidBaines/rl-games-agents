from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallBoardCNN(BaseFeaturesExtractor):
    """
    A compact CNN that works well for small square boards (e.g., 16x16).

    Accepts observations in CHW format and produces a dense feature vector.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        # observation_space should be (C, H, W)
        super().__init__(observation_space, features_dim)
        assert len(observation_space.shape) == 3, "Expected CHW observations"
        channels, height, width = observation_space.shape

        # Convolutional stack keeps spatial dims valid for small inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
        )

        # Compute the flattened size by running a dummy forward
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, channels, height, width)).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:  # type: ignore[override]
        return self.linear(self.cnn(observations))


