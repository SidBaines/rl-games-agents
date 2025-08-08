# RL Board Games

Reinforcement learning agents for board games, starting with Ricochet Robots.

## Installation

```bash
# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Usage

```python
from rl_board_games.games.ricochet_robots import RicochetRobotsGame, AStarSolver

# Create a game
game = RicochetRobotsGame()
state = game.reset(seed=42)

# Solve with A*
solver = AStarSolver(game)
actions = solver.solve(state)
print(f"Solution found in {len(actions)} moves")
```

## Testing

```bash
pytest
``` 