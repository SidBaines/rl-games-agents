Removed duplicate plan move bounds from `PlanCurriculumLevel`.

- Eliminated `max_total_moves` and `min_total_moves` from `PlanCurriculumLevel` in `rl_board_games/curricula/astar_plan_curriculum.py`.
- Updated logic to rely solely on `min_solve_length`/`max_solve_length` for total move constraints.
- Adjusted cache sampling/predicates and validation accordingly.
- Updated `scripts/generate_plan_cache.py`, `scripts/train_curriculum.py`, and `notebooks/visualise_astar_plan_curriculum.py` to drop the removed fields and use solve length bounds.
- Left config YAMLs unchanged for now; any `min_total_moves`/`max_total_moves` entries are ignored by the updated code.

Notebook updates:
- `notebooks/visualise_astar_plan_curriculum.py` can now optionally load an A* plan curriculum from a YAML config via `CONFIG_PATH`; falls back to defaults when unset/invalid.

Offline metrics & plots when W&B is disabled:
- Added offline metric collection to `rl_board_games/training/trainer.py`. When `use_wandb=False`, evaluation metrics are stored and exported to `tmp/<datetime>/` as `metrics.json`, `metrics.csv`, and plotted to `metrics.png`/`metrics.pdf`.
- Wired `scripts/train_curriculum.py` to pass a `run_dir` for offline runs.
- Added `matplotlib` to `requirements.txt`.
 - Persist metrics incrementally after each evaluation and export plots again on `Trainer.close()` as a fallback. Also print the offline run directory at startup.
 - Generate/overwrite offline plots at each evaluation interval so they update live during training.


CNN policy support:
- Added `policy` parameter to SB3 agents (`rl_board_games/agents/sb3/{ppo_agent.py,dqn_agent.py}`) to allow selecting `CnnPolicy` or `MlpPolicy`.
- Training script now infers a default policy based on encoder type (`scripts/train_curriculum.py`): `CnnPolicy` for `planar`/`rgb`, `MlpPolicy` otherwise; can be overridden via `agent.policy` in YAML.
- New config `configs/ricochet_robots/ppo_cnn_curriculum.yaml` demonstrating PPO with `CnnPolicy` and image encoders.
- Added `policy_kwargs` plumbed through, and set `normalize_images=False` by default for image encoders to avoid double-normalization with SB3.
