# Ricochet Robots RNG determinism: structured board generation

- Problem: Both curriculum and cache generation consumed `rng.randint(...)` for `num_walls` before calling `Board.random_walls`. For structured boards (size>=8 and even), `Board.random_walls` ignores `num_walls` but uses `rng` heavily. Python's `randrange`/`getrandbits` can consume a variable number of internal draws depending on bounds. If `max_walls` differs across configs, the pre-draw advances RNG by a different amount, producing different boards for the same seed.
- Change: Avoid any RNG consumption prior to structured board generation. Only draw `num_walls` for small/odd sizes where the legacy random wall generator actually uses it.
- Files:
  - `rl_board_games/curricula/astar_plan_curriculum.py` (`_create_game_from_seed`)
  - `scripts/generate_plan_cache.py` (`create_game_from_seed`)
- Impact: Same seed + same size now yields identical boards across configs; cached seeds and runtime generation align. Slight behavior change only for small/odd sizes (unchanged from previous intent). 