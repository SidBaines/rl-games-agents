# Ricochet Robots: Structured Wall Generator (2025-08-11)

- Implemented `Board.ricochet_walls(size, N=2, M=4, rng)` in `rl_board_games/games/ricochet_robots/board.py`.
- Updated `Board.random_walls(...)` to delegate to `ricochet_walls` when `size >= 8` and even; otherwise, legacy random placement is used as fallback.
- Rules implemented:
  - Central 2x2 grid is enclosed by walls.
  - Per quarter (NW, NE, SW, SE): place N edge spurs extending inward from the board edge, never on the exact midline.
  - Per quarter: place up to M L-shaped wall pairs, with orientations distributed as evenly as possible; Ls do not touch each other, the central 2x2 perimeter, or edge spurs.
- All randomness is driven solely by the provided `rng` instance to ensure reproducibility by seed across the entire pipeline (curricula and scripts reuse the same seed to construct the `rng`).
- Existing tests pass (`25 passed, 1 warning`).

Notes/TODO:
- The exact RR placement rules can be refined. The current implementation prioritizes reproducibility and separable rule units, making it easy to adjust constraints (e.g., minimum spacing, orientation distributions, spur definitions).
- `num_walls` is ignored for structured generation by design; future extension could expose global density controls. 