# %% [markdown]
"""
# A* Plan Curriculum Visualisation

Visualise the A* plan-based curriculum levels, sample boards, A* solutions
(number of moves and number of robots moved), and environment transitions.

Run in Jupyter or as a script. Requires matplotlib.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

from rl_board_games.games.ricochet_robots import (
    RicochetRobotsGame,
    WallAwarePlanarEncoder,
    AStarSolver,
)
from rl_board_games.training.curriculum_env import CurriculumRicochetRobotsEnv
from rl_board_games.curricula.astar_plan_curriculum import AStarPlanCurriculum, PlanCurriculumLevel

# %% [markdown]
"""
## Optional: configure curriculum via YAML

Set `CONFIG_PATH` to a YAML config to build the A* curriculum from file. Leave as `None`
to use the built-in defaults.
"""

# %%
from pathlib import Path
import yaml

# CONFIG_PATH: str | None = None  # e.g., "configs/ricochet_robots/ppo_astar_curruculum2.yaml"
# CONFIG_PATH: str | None = "/Users/sidbaines/Documents/Code/20250716_tmp/configs/ricochet_robots/ppo_cnn_curriculum.yaml"
CONFIG_PATH: Optional[str] = "../configs/ricochet_robots/ppo_cnn_curriculum.yaml"


def build_astar_curriculum_from_config(config_path: Optional[str]) -> AStarPlanCurriculum:
    if not config_path:
        return AStarPlanCurriculum()

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}, falling back to defaults")
        return AStarPlanCurriculum()

    with cfg_path.open() as f:
        config = yaml.safe_load(f)

    curriculum_cfg = (config or {}).get("curriculum", {})
    if curriculum_cfg.get("type") != "astar_plan":
        print("Config curriculum.type is not 'astar_plan'; using defaults")
        return AStarPlanCurriculum()

    levels: list[PlanCurriculumLevel] = []
    for level_cfg in curriculum_cfg.get("levels", []):
        level = PlanCurriculumLevel(
            name=level_cfg["name"],
            min_solve_length=level_cfg["min_solve_length"],
            max_solve_length=level_cfg["max_solve_length"],
            success_threshold=level_cfg["success_threshold"],
            board_size=level_cfg["board_size"],
            num_robots=level_cfg["num_robots"],
            max_walls=level_cfg["max_walls"],
            episodes_per_evaluation=level_cfg["episodes_per_evaluation"],
            board_size_min=level_cfg.get("board_size_min"),
            board_size_max=level_cfg.get("board_size_max"),
            max_robots_moved=level_cfg.get("max_robots_moved", 1),
            min_robots_moved=level_cfg.get("min_robots_moved", 1),
            max_episode_steps=level_cfg.get("max_episode_steps"),
        )
        levels.append(level)

    return AStarPlanCurriculum(
        levels=levels or None,
        evaluation_episodes=curriculum_cfg.get("evaluation_episodes", 50),
        plan_cache_dir=curriculum_cfg.get("plan_cache_dir", "plan_lookup"),
    )

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


def plan_stats(plan) -> tuple[int, int]:
    total_moves = len(plan) if plan is not None else 0
    robots_used = len({r for (r, _d) in plan}) if plan else 0
    return total_moves, robots_used


# %% [markdown]
"""
## Visualise a few samples from current level
"""

# %%
# Build A* plan curriculum (optionally from CONFIG_PATH)
curriculum = build_astar_curriculum_from_config(CONFIG_PATH)
encoder = WallAwarePlanarEncoder()

# Peek at first N games from curriculum (using current level)
N = 1
samples: List[RicochetRobotsGame] = []
iter_cur = iter(curriculum)
for i in range(N):
    game = next(iter_cur)
    # Reset deterministically to the seed used during generation
    seed = getattr(game, "initial_seed", None)
    state = game.reset(seed=seed)
    samples.append(game)

    # Render and solve with A*
    img = game.render(state, mode="rgb_array")
    show_rgb(img, title=f"Sample {i+1}: {curriculum.get_current_level().name} ({game.board.width}x{game.board.height})")




# %%
# Build A* plan curriculum (optionally from CONFIG_PATH)
curriculum = build_astar_curriculum_from_config(CONFIG_PATH)
encoder = WallAwarePlanarEncoder()

# Peek at first N games from curriculum (using current level)
N = 1
samples: List[RicochetRobotsGame] = []
iter_cur = iter(curriculum)
for i in range(N):
    game = next(iter_cur)
    # Reset deterministically to the seed used during generation
    seed = getattr(game, "initial_seed", None)
    state = game.reset(seed=seed)
    samples.append(game)

    # Render and solve with A*
    img = game.render(state, mode="rgb_array")
    show_rgb(img, title=f"Sample {i+1}: {curriculum.get_current_level().name} ({game.board.width}x{game.board.height})")

    solver = AStarSolver(game, max_depth=50)
    plan = solver.solve(state)
    total_moves, robots_used = plan_stats(plan)
    print(f"Sample {i+1}: plan length={total_moves}, robots moved={robots_used}, board_sig={state.board_sig}")

    # Step through plan and render final state
    current_state = state
    game._state = current_state
    for act in (plan or []):
        current_state, _, done, _ = game.step(act)
        if done:
            break
    show_rgb(game.render(current_state, mode="rgb_array"), title=f"Sample {i+1}: After A* plan ({total_moves} moves)")

# %% [markdown]
"""
## Environment transitions: reset → step → (done/truncated)

Observe state changes in a CurriculumRicochetRobotsEnv driven by the A* plan curriculum.
"""

# %%
# Create environment
env = CurriculumRicochetRobotsEnv(curriculum=curriculum, encoder=encoder, max_episode_steps=30, curriculum_update_freq=1)
render_state(env.game, title="Env before reset")
print("Environment created. About to reset...")

# Reset
obs, info = env.reset()
print("Reset info:", info)
print("Obs shape:", obs.shape)
render_state(env.game, title="Env after reset")
obs, info = env.reset()
render_state(env.game, title="Env after second reset")

# Compute A* plan on the reset state and show stats
solver = AStarSolver(env.ricochet_game, max_depth=50)
state_for_plan = env.game._state
plan = solver.solve(state_for_plan)
moves, robots = plan_stats(plan)
print(f"Initial A* plan: moves={moves}, robots moved={robots}")

# Step through a few actions (use plan if available, otherwise random)
num_steps = min(5, moves) if plan else 3
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

# If finished, reset again to see a new curriculum sample
if done or truncated:
    obs, info = env.reset()
    print("Post-episode reset info:", info)
    render_state(env.game, title="Env after post-episode reset")

# %% [markdown]
"""
## Level sweep demo

Visualize one sample per level, reporting A* plan length and number of robots moved.
"""

# %%
levels = curriculum.levels
print("A* Plan Curriculum levels:")
for lvl in levels:
    print(f"- {lvl.name}: size {lvl.board_size_min or lvl.board_size}..{lvl.board_size_max or lvl.board_size}, max_solve_length={getattr(lvl, 'max_solve_length', '?')}, max_robots_moved={getattr(lvl, 'max_robots_moved', '?')}")

print("\nSampling one game per level:")
for lvl in levels:
    # Force current level
    curriculum.state.current_level = levels.index(lvl)
    game = next(iter(curriculum))
    st = game.reset(seed=getattr(game, "initial_seed", None))
    show_rgb(game.render(st, mode="rgb_array"), title=f"Level {lvl.name}")
    plan = AStarSolver(game, max_depth=50).solve(st)
    total_moves, robots_used = plan_stats(plan)
    print(f"Level {lvl.name}: plan length={total_moves}, robots moved={robots_used}, board_sig={st.board_sig}")

# %%
print("Done.") 
# %%
