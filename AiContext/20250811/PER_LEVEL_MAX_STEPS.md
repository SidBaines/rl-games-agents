Per-level max episode steps support

- Added optional `max_episode_steps` to `CurriculumLevel` to allow per-level episode caps.
- Updated `CurriculumRicochetRobotsEnv` to apply level-specific `max_episode_steps` when switching levels, with fallback to the environment default.
- Wired YAML-defined A* plan curriculum levels into `AStarPlanCurriculum` and included support for `max_episode_steps` in those levels.

Files edited:
- `rl_board_games/core/curriculum.py`
- `rl_board_games/training/curriculum_env.py`
- `scripts/train_curriculum.py`

How to use:
- In your curriculum YAML, set `max_episode_steps` under each level to override the episode limit for that level. If omitted, the top-level `environment.max_episode_steps` is used.

Notes:
- For `curriculum.type: astar_plan`, YAML levels are now respected (previously defaults were always used). Include `max_total_moves`, `max_robots_moved`, and optional `board_size_min`/`board_size_max` as needed. 