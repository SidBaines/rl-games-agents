# A* Plan Curriculum: Cached-seed revalidation

- Changed `AStarPlanCurriculum._generate_game_for_level` to re-run A* and re-check constraints when a seed is retrieved from the plan cache before returning it.
- Rationale: seeds in the cache are keyed by `(board_size, num_robots, seed)` but not by `max_walls`. Using the same seed under a different `max_walls` (or other generation params) can produce a different board that violates the level constraints (e.g., not 1-move solvable). Revalidation guarantees correctness.
- Impact: Slight extra solve cost the first time a cached seed is used per session, but prevents serving invalid boards and avoids surprising visualizations.
- Alternatives: include `max_walls` (and/or `num_walls`) in the cache key and sampling; use distinct `plan_cache_dir` per config; or disable the cache during debugging. 