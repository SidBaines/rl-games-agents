- Changes:
  - Tightened `R1-L1` level to require exactly one move and one robot (`min_total_moves=1`, `min_robots_moved=1`) in `rl_board_games/curricula/astar_plan_curriculum.py`.
  - Prevented zero-length plans from being written to the plan cache in both the curriculum and the cache generator.
  - Increased default A* solver depth in `scripts/generate_plan_cache.py` from 15 to 50 to align with curriculum and reduce false 0-move entries.

- Impact:
  - `R1-L1` now consistently yields single-move, single-robot solutions.
  - Cache no longer accumulates `moves0/robots0` combinations, avoiding sampling invalid entries.

- Files touched:
  - `rl_board_games/curricula/astar_plan_curriculum.py`
  - `scripts/generate_plan_cache.py` 