Change: Ensure the goal is always placed on an L-square.

- Added Board.l_centers() to compute all cells with exactly two adjacent walls (N+E, E+S, S+W, W+N).
- Updated RicochetRobotsGame.reset() to choose the goal uniformly from L-centers excluding robot positions, with fallback to a random empty cell if none exist.
- Verified via smoke test: generated structured board has L-centers; goal chosen lies in L-centers.

Rationale: This is efficient (single pass over wall grid) and keeps responsibility well-separated: board provides structure queries; game uses them for target placement.
