# %% [markdown]
"""
# Curriculum & Environment Visualisation

This developer notebook helps verify:
- Curriculum levels generation (boards, robots, goals) across difficulties
- A* solutions per generated game
- Environment transitions: before/after reset, step, and truncation

Run in Jupyter or as a script. Requires matplotlib and ipywidgets for interactive bits.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from typing import List

from rl_board_games.games.ricochet_robots import (
    RicochetRobotsGame,
    WallAwarePlanarEncoder,
    AStarSolver,
)
from rl_board_games.training.curriculum_env import CurriculumRicochetRobotsEnv
from rl_board_games.curricula.ricochet_robots_curriculum import RicochetRobotsCurriculum
from rl_board_games.core.curriculum import CurriculumLevel, DifficultyLookup

# %% [markdown]
"""
## Helpers
"""

# %%

def show_rgb(img: np.ndarray, title: str = "") -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def render_state(game: RicochetRobotsGame, title: str = "") -> None:
    show_rgb(game.render(None, mode="rgb_array") if game._state is None else game.render(game._state, mode="rgb_array"), title)


# %% [markdown]
"""
## Visualise a few curriculum levels and their A* solutions
"""

# %%
# Build curriculum with defaults and a DifficultyLookup (may be empty; then it falls back to random)
lookup = DifficultyLookup("difficulty_lookup")
curriculum = RicochetRobotsCurriculum(difficulty_lookup=lookup)

# Peek at first N games from curriculum (using current level)
N = 3
encoder = WallAwarePlanarEncoder()

samples: List[RicochetRobotsGame] = []
iter_cur = iter(curriculum)
for i in range(N):
    game = next(iter_cur)
    state = game.reset(seed=getattr(game, "initial_seed", None))
    samples.append(game)

    # Render and solve with A*
    img = game.render(state, mode="rgb_array")
    show_rgb(img, title=f"Sample {i+1}: {curriculum.get_current_level().name} ({game.board.width}x{game.board.height})")

    solver = AStarSolver(game, max_depth=50)
    plan = solver.solve(state)
    print(f"Sample {i+1} plan length: {len(plan)} | board_sig={state.board_sig}")

    # Step through plan and render final state
    current_state = state
    game._state = current_state
    for act in plan:
        current_state, _, done, _ = game.step(act)
        if done:
            break
    show_rgb(game.render(current_state, mode="rgb_array"), title=f"Sample {i+1}: After A* plan ({len(plan)} moves)")

# %% [markdown]
"""
## Environment transitions: reset → step → (done/truncated)

We use CurriculumRicochetRobotsEnv to observe state changes at key points.
- Before reset: no state
- After reset: curriculum may swap game; observation reflects encoder output
- After step: observe reward/done/truncated and rendering
"""

# %%
# Create environment
env = CurriculumRicochetRobotsEnv(curriculum=curriculum, encoder=encoder, max_episode_steps=30, curriculum_update_freq=1)

# Before reset: nothing to render (env holds the game but no state yet)
print("Environment created. About to reset...")

# Print the A* solver max steps before and after reset
solver = AStarSolver(env.game, max_depth=50)
plan = solver.solve(env.game._state)
render_state(env.game, title=f"Env before reset, A* max steps: {len(plan)}")

# Reset
obs, info = env.reset()
solver = AStarSolver(env.game, max_depth=50)
plan = solver.solve(env.game._state)
print("Reset info:", info)
print("Obs shape:", obs.shape)
render_state(env.game, title=f"Env after reset, A* max steps: {len(plan)}")

# Take a few steps with a simple heuristic: follow first A* action if available
solver = AStarSolver(env.ricochet_game, max_depth=50)
state_for_plan = env.game._state
plan = solver.solve(state_for_plan)
print("Initial plan length:", len(plan))

num_steps = min(5, len(plan)) if plan else 3
for t in range(num_steps):
    if plan and t < len(plan):
        action = env._encode_action(*plan[t])
    else:
        action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {t+1}: reward={reward}, done={done}, truncated={truncated}")
    render_state(env.game, title=f"Env after step {t+1}")
    if done or truncated:
        break

# If episode finished, reset again to see a new curriculum sample
if done or truncated:
    obs, info = env.reset()
    print("Post-episode reset info:", info)
    render_state(env.game, title="Env after post-episode reset")

# %% [markdown]
"""
## Level sweep demo

Try a representative set of levels (if lookup has data) or fall back to defaults
and visualize generated boards quickly.
"""

# %%
levels = curriculum.levels
print("Curriculum levels:")
for lvl in levels:
    print(f"- {lvl.name}: size={lvl.board_size} robots={lvl.num_robots} target moves {lvl.min_solve_length}-{lvl.max_solve_length}")

print("\nSampling one game per level:")
for lvl in levels:
    # Temporarily force current level by directly setting state
    curriculum.state.current_level = levels.index(lvl)
    game = next(iter(curriculum))
    st = game.reset(seed=getattr(game, "initial_seed", None))
    show_rgb(game.render(st, mode="rgb_array"), title=f"Level {lvl.name} ({lvl.board_size}x{lvl.board_size}, robots={lvl.num_robots})")
    plan = AStarSolver(game, max_depth=50).solve(st)
    print(f"Level {lvl.name}: plan length={len(plan)} board_sig={st.board_sig}")

# %%
print("Done.") 
# %%
