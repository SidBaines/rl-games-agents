# Repository review notes – 2025-08-08

## Glaring/obvious issues found
- PlanarEncoder produced inconsistent shapes and omitted wall planes.
- Difficulty generator file was corrupted with stray characters/newlines, breaking execution.
- Trainer consumed `eval/num_episodes` but did not include it in returned metrics.
- `RRGameState` instances were used as dict keys in A* (came_from/g_cost) without explicit hash control; `Board` reference made hashing/equality heavy and potentially error-prone.

## Edits applied
- Updated `games/ricochet_robots/encoders.py` PlanarEncoder to use `state.board.height/width` and add 4 wall planes.
- Rewrote `core/difficulty_generator.py` cleanly; same API/CLI but fixed logic and formatting.
- Added `eval/num_episodes` to metrics in `training/trainer.py` and used underlying SB3 model for evaluation actions when available.
- Made `RRGameState.board` excluded from dataclass equality/hash (`field(compare=False, hash=False, repr=False)`) so states are lightweight hashable for A*.
- Added `Board.signature()` (64-bit BLAKE2b over dimensions + wall bytes) and included `board_sig` in `RRGameState` hashing, keeping `board` excluded to keep equality fast but walls-aware.

## Follow-ups / suggestions
- Consider unit tests for encoders’ shapes and wall planes.
- Add CI to run tests and lint.
- Benchmark `Board.next_position` with Numba if hotspot.
- Ensure WandB optional import handling to avoid import failures when disabled. 

## Additional notes (determinism & curricula)
- Curriculum resets: the env may swap in a new game on reset per iterator cadence. Resets are deterministic for a given game (`initial_seed`), but not across different games unless you freeze the iterator. For debugging determinism, consider a freeze-iterator option.
- Level board sizes: updated levels to sample `board_size` in a small range per level to avoid overfitting to a single size, while keeping difficulty bounds consistent via lookup/fallback.
- New curriculum: added `AStarPlanCurriculum` (`rl_board_games/curricula/astar_plan_curriculum.py`) using A* plan constraints (`max_total_moves`, `max_robots_moved`). Supports optional board size ranges per level and exposes seeds for deterministic per-game resets. `scripts/train_curriculum.py` can switch via YAML `curriculum.type` (`ricochet_robots` | `astar_plan`). 