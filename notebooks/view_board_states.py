# %% [markdown]
"""
# Board State Visualisation Demo

Use this notebook to visualise Ricochet Robots board states and see agents act on them.
"""

# %%
from rl_board_games.games.ricochet_robots import RicochetRobotsGame, FlatArrayEncoder, WallAwarePlanarEncoder, AStarSolver
from rl_board_games.games.ricochet_robots.board import Board, NORTH, EAST, SOUTH, WEST
from rl_board_games.agents.heuristic import RandomAgent

import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets

# Create a small board and game
board = Board.random_walls(size=8, num_walls=12)

game = RicochetRobotsGame(board=board, num_robots=4)
state = game.reset(seed=0)

# Encoders
flat_encoder = FlatArrayEncoder()
planar_encoder = WallAwarePlanarEncoder(board)

# %% [markdown]
"""
## Render modes
"""

# %%
print("Human render:\n")
game.render(state, mode="human")

# %%
plt.figure(figsize=(4,4))
plt.imshow(game.render(state, mode="rgb_array"))
plt.title("RGB Array Render")
plt.axis('off')
plt.show()

# %% [markdown]
"""
## Interactive agent playback
"""

# %%
solver = AStarSolver(game, max_depth=20)
agent = RandomAgent(game)

# Generate a solution plan using A*
plan = solver.solve(state)
print("A* plan length:", len(plan))

# %%
# Widget to step through actions
step_slider = widgets.IntSlider(value=0, min=0, max=len(plan), description='Step')
output = widgets.Output()

@widgets.interact(step=step_slider)
def show_step(step):
    output.clear_output(wait=True)
    with output:
        current_state = state
        game._state = current_state  # sync internal state
        for i in range(step):
            current_state, _, _, _ = game.step(plan[i])
        plt.figure(figsize=(4,4))
        plt.imshow(game.render(current_state, mode="rgb_array"))
        plt.title(f"Step {step}")
        plt.axis('off')
        plt.show()

output 
# %%
