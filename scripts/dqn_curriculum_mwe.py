# %% [markdown]
"""
# DQN Curriculum MWE

Minimal working example (MWE) that trains a DQN agent on the Ricochet Robots
game using a simple curriculum. The curriculum advances through difficulty
levels defined by the *minimum* A* solution length of randomly-generated
boards.

This file is a *jupyter-cell-tagged* python script, so you can open it in
JupyterLab or VS Code and run cell-by-cell. Execute the whole file as a normal
script as well:

```bash
python scripts/dqn_curriculum_mwe.py
```
"""

# %%
# Standard library
import itertools
import random
from pathlib import Path
from typing import Iterator, List

# Third-party
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Project imports – make sure project root is on sys.path
import sys
from pathlib import Path as _Path

project_root = _Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_board_games.games.ricochet_robots import (
    RicochetRobotsGame,
    FlatArrayEncoder,
    AStarSolver,
)
from rl_board_games.training.ricochet_robots_env import RicochetRobotsEnv
from rl_board_games.games.ricochet_robots.board import Board

# %% [markdown]
"""
## Curriculum helper

The curriculum yields games whose shortest A* solution length is at least the
requested *difficulty*. We keep sampling random boards until one satisfies the
criterion.  In a production set-up you would cache boards or generate them
offline, but for this demo on small boards it is fast enough.
"""

# %%

def generate_game_with_min_astar(min_len: int, *, size: int = 8, seed: int | None = None) -> RicochetRobotsGame:
    """Return a game whose optimal A* solution length >= ``min_len``."""
    rng = random.Random(seed)
    while True:
        board = Board.random_walls(size=size, num_walls=rng.randint(5, 15), rng=rng)
        game = RicochetRobotsGame(board=board, num_robots=4, rng=rng)
        solver = AStarSolver(game, max_depth=30)
        solution = solver.solve(game.reset())
        if solution and len(solution) >= min_len:
            return game


class CurriculumIterator(Iterator[RicochetRobotsGame]):
    """Endless iterator over games, cycling through *difficulties*."""

    def __init__(self, difficulties: List[int], seed: int | None = None):
        self.difficulties = difficulties
        self.rng = random.Random(seed)
        self._iter = self._make_iter()

    def _make_iter(self):
        while True:
            for d in self.difficulties:
                yield generate_game_with_min_astar(d, seed=self.rng.randint(0, 1 << 32))

    def __next__(self) -> RicochetRobotsGame:
        return next(self._iter)

    def __iter__(self):
        return self


# %% [markdown]
"""
## Environment wrapper that pulls a fresh game from the curriculum on every reset
"""

# %%

class CurriculumRicochetEnv(RicochetRobotsEnv):
    def __init__(self, curriculum: Iterator[RicochetRobotsGame]):
        self.curriculum = curriculum
        # Build initial game / encoder
        self.current_game = next(self.curriculum)
        encoder = FlatArrayEncoder()
        super().__init__(self.current_game, encoder, max_episode_steps=50)

    # Override reset so that each episode draws a new game matching curriculum
    def reset(self, seed: int | None = None, **kwargs):
        self.current_game = next(self.curriculum)
        # Swap underlying game reference
        self.ricochet_game = self.current_game
        self.game = self.current_game
        return super().reset(seed=seed, **kwargs)


# %% [markdown]
"""
## Metric logging helper

Evaluates the policy on a fixed set of probe difficulties and returns a metric
dict.  We evaluate for *n_eval_episodes* per level.
"""

# %%

def evaluate_on_difficulties(model: DQN, difficulties: List[int], n_eval_episodes: int = 3) -> dict:
    encoder = FlatArrayEncoder()
    rewards_by_level = {}
    for d in difficulties:
        envs = []
        for _ in range(n_eval_episodes):
            game = generate_game_with_min_astar(d, seed=None)
            env = RicochetRobotsEnv(game, encoder, max_episode_steps=50)
            envs.append(env)
        total = 0
        for env in envs:
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < env.max_episode_steps:
                action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
                obs, reward, done, truncated, _ = env.step(int(action))
                total += reward
                steps += 1
        rewards_by_level[d] = total / n_eval_episodes
    # Flatten for logging
    return {f"eval/mean_reward_len_{k}": v for k, v in rewards_by_level.items()}


# %% [markdown]
"""
## Training loop parameters
"""

# %%
TOTAL_TIMESTEPS = 10_000  # Keep small for demo
EVAL_INTERVAL = 2_000
SAVE_PATH = Path("demo_checkpoints/dqn_curriculum_mwe.zip")
DIFFICULTIES = [3, 5, 7, 9]


# %% [markdown]
"""
## Build environment & agent
"""

# %%

curriculum = CurriculumIterator(DIFFICULTIES, seed=42)
train_env = DummyVecEnv([lambda: CurriculumRicochetEnv(curriculum)])
model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    batch_size=32,
    buffer_size=10_000,
    learning_starts=1_000,
    gamma=0.99,
    verbose=1,
)


# %% [markdown]
"""
## Training with periodic evaluation
"""

# %%

timesteps = 0
while timesteps < TOTAL_TIMESTEPS:
    next_chunk = min(EVAL_INTERVAL, TOTAL_TIMESTEPS - timesteps)
    model.learn(total_timesteps=next_chunk, reset_num_timesteps=False)
    timesteps += next_chunk
    metrics = evaluate_on_difficulties(model, DIFFICULTIES, n_eval_episodes=2)
    metrics["timestep"] = timesteps
    print("Metrics:", metrics)

# Save final model
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(SAVE_PATH)
print(f"Model saved to {SAVE_PATH.relative_to(project_root)}")


# %% [markdown]
"""
## Quick test – load model and run one episode on hardest level
"""

# %%

if __name__ == "__main__":
    # Load back the model just saved
    loaded_model = DQN.load(SAVE_PATH, env=train_env)
    test_game = generate_game_with_min_astar(DIFFICULTIES[-1])
    test_env = RicochetRobotsEnv(test_game, FlatArrayEncoder(), max_episode_steps=50)
    obs, _ = test_env.reset()
    done = False
    steps = 0
    while not done and steps < test_env.max_episode_steps:
        action, _ = loaded_model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, done, truncated, _ = test_env.step(int(action))
        steps += 1
    print(f"Finished episode in {steps} steps, done={done} truncated={truncated}") 