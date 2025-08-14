from __future__ import annotations

import random
import hashlib
from typing import List, Tuple, Iterable, Set

import numpy as np

# Direction indices
NORTH, EAST, SOUTH, WEST = range(4)
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]
# Bitmask encoding for walls: 1<<direction
DIR_MASKS = [1 << d for d in range(4)]
OPPOSITE_DIR = {NORTH: SOUTH, EAST: WEST, SOUTH: NORTH, WEST: EAST}


class Board:
    """Light-weight board with immutable walls and size."""

    def __init__(self, width: int = 16, height: int | None = None, walls: np.ndarray | None = None):
        self.width = int(width)
        self.height = int(height) if height is not None else int(width)
        if walls is None:
            self.walls = np.zeros((self.height, self.width), dtype=np.uint8)
            self._add_board_boundaries()
        else:
            assert walls.shape == (self.height, self.width)
            self.walls = walls.astype(np.uint8)

    # ---------------------------------------------------------------------
    # Wall helpers
    # ---------------------------------------------------------------------
    def _add_board_boundaries(self) -> None:
        # North & South outer walls
        self.walls[0, :] |= DIR_MASKS[NORTH]
        self.walls[-1, :] |= DIR_MASKS[SOUTH]
        # West & East outer walls
        self.walls[:, 0] |= DIR_MASKS[WEST]
        self.walls[:, -1] |= DIR_MASKS[EAST]

    def add_wall(self, x: int, y: int, direction: int) -> None:
        """Add a wall on cell (x,y) for the given direction and the opposite on the adjacent cell."""
        self.walls[y, x] |= DIR_MASKS[direction]
        nx, ny = x + DX[direction], y + DY[direction]
        if 0 <= nx < self.width and 0 <= ny < self.height:
            self.walls[ny, nx] |= DIR_MASKS[OPPOSITE_DIR[direction]]

    def has_wall(self, x: int, y: int, direction: int) -> bool:
        return bool(self.walls[y, x] & DIR_MASKS[direction])

    def signature(self) -> int:
        """Return a stable, compact integer signature for walls and size.

        Uses BLAKE2b on the walls bytes and dimensions to avoid collisions
        across different boards while keeping hashing lightweight.
        """
        hasher = hashlib.blake2b(digest_size=8)
        # Include dimensions explicitly
        hasher.update(self.width.to_bytes(2, byteorder="little", signed=False))
        hasher.update(self.height.to_bytes(2, byteorder="little", signed=False))
        # Include wall bytes
        hasher.update(self.walls.tobytes())
        return int.from_bytes(hasher.digest(), byteorder="little", signed=False)

    # ---------------------------------------------------------------------
    # Movement helpers
    # ---------------------------------------------------------------------
    def next_position(
        self,
        x: int,
        y: int,
        direction: int,
        robots: List[Tuple[int, int]] | Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, int]:
        """Return the position where a robot will stop given a direction."""
        width, height = self.width, self.height
        robot_set = set(robots)
        cx, cy = x, y
        while True:
            if self.has_wall(cx, cy, direction):
                break
            nx, ny = cx + DX[direction], cy + DY[direction]
            if not (0 <= nx < width and 0 <= ny < height):
                # Shouldn't happen due to boundary walls, but safe-guard.
                break
            if (nx, ny) in robot_set:
                break
            cx, cy = nx, ny
        return cx, cy

    def l_centers(self) -> List[Tuple[int, int]]:
        """Return positions that form an L (exactly two orthogonal walls on the cell).

        A cell is considered an L-center if and only if it has exactly two
        walls and they are adjacent directions (N+E, E+S, S+W, or W+N).
        """
        centers: List[Tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                mask = int(self.walls[y, x])
                if mask == 0:
                    continue
                has_n = bool(mask & DIR_MASKS[NORTH])
                has_e = bool(mask & DIR_MASKS[EAST])
                has_s = bool(mask & DIR_MASKS[SOUTH])
                has_w = bool(mask & DIR_MASKS[WEST])
                count = int(has_n) + int(has_e) + int(has_s) + int(has_w)
                if count != 2:
                    continue
                if (has_n and has_e) or (has_e and has_s) or (has_s and has_w) or (has_w and has_n):
                    centers.append((x, y))
        return centers

    # ---------------------------------------------------------------------
    # Factory helpers
    # ---------------------------------------------------------------------
    @classmethod
    def empty(cls, size: int = 16) -> "Board":
        return cls(width=size, height=size)

    @classmethod
    def random_walls(cls, size: int = 16, num_walls: int = 20, rng: random.Random | None = None) -> "Board":
        """Generate a board with walls.

        If the board is square, even-sized, and at least 8x8, generate walls
        using a structured Ricochet Robots layout with seeded randomness.
        Otherwise, fall back to simple random single-segment walls.

        Note: the num_walls parameter is ignored for structured generation.
        """
        rng = rng or random.Random()
        if size >= 8 and size % 2 == 0:
            return cls.ricochet_walls(size=size, rng=rng)
        # Fallback legacy random walls for small/odd sizes
        board = cls.empty(size)
        for _ in range(num_walls):
            x = rng.randrange(1, size - 1)
            y = rng.randrange(1, size - 1)
            direction = rng.choice([NORTH, EAST, SOUTH, WEST])
            board.add_wall(x, y, direction)
        return board

    # ---------------------------------------------------------------------
    # Structured Ricochet Robots layout
    # ---------------------------------------------------------------------
    @classmethod
    def ricochet_walls(
        cls,
        size: int = 16,
        N: int = 2,
        M: int = 4,
        rng: random.Random | None = None,
    ) -> "Board":
        """Generate a RR-style board layout following supplied rules.

        Rules implemented (loosely, with safeguards):
        0) If size < 8, odd, or non-square, falls back to legacy random walls.
        1) Central 2x2 grid has walls around it.
        2) In each quarter, place exactly one edge spur on each of its two edges
           (e.g., NW: one on the north edge and one on the west edge).
        3) In each quarter, place up to M L-shaped internal wall pairs, as evenly
           distributed across the 4 orientations as possible, without touching
           other L-pairs, edge spurs, or the central 2x2 perimeter.

        Seeding: All random choices are driven only by the provided rng.
        """
        rng = rng or random.Random()
        if size < 8 or size % 2 != 0:
            # Fallback to legacy behavior
            return cls.random_walls(size=size, num_walls=20, rng=rng)

        board = cls.empty(size)

        # Helpers -----------------------------------------------------------
        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < size and 0 <= y < size

        def chebyshev_neighbors(x: int, y: int) -> Iterable[Tuple[int, int]]:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny):
                        yield (nx, ny)

        def manhattan_neighbors(x: int, y: int) -> Iterable[Tuple[int, int]]:
            for d in (NORTH, EAST, SOUTH, WEST):
                nx, ny = x + DX[d], y + DY[d]
                if in_bounds(nx, ny):
                    yield (nx, ny)

        mid1 = size // 2 - 1
        mid2 = size // 2

        # 1) Central 2x2 perimeter walls -----------------------------------
        # Cells: (mid1, mid1) (mid2, mid1)
        #        (mid1, mid2) (mid2, mid2)
        board.add_wall(mid1, mid1, NORTH)
        board.add_wall(mid1, mid1, WEST)
        board.add_wall(mid2, mid1, NORTH)
        board.add_wall(mid2, mid1, EAST)
        board.add_wall(mid1, mid2, SOUTH)
        board.add_wall(mid1, mid2, WEST)
        board.add_wall(mid2, mid2, SOUTH)
        board.add_wall(mid2, mid2, EAST)

        # Build a forbidden-zone set around the central 2x2 to prevent touching
        central_cells = {(mid1, mid1), (mid2, mid1), (mid1, mid2), (mid2, mid2)}
        central_forbidden: Set[Tuple[int, int]] = set()
        for cx, cy in central_cells:
            central_forbidden.add((cx, cy))
            central_forbidden.update(chebyshev_neighbors(cx, cy))

        # Quarter bounding boxes (inclusive ranges)
        # NW, NE, SW, SE
        quarters = {
            "NW": (range(0, mid2), range(0, mid2)),
            "NE": (range(mid2, size), range(0, mid2)),
            "SW": (range(0, mid2), range(mid2, size)),
            "SE": (range(mid2, size), range(mid2, size)),
        }

        # Precompute edge cells (board boundary cells have walls)
        edge_cells: Set[Tuple[int, int]] = set()
        for y in range(size):
            for x in range(size):
                if x == 0 or y == 0 or x == size - 1 or y == size - 1:
                    edge_cells.add((x, y))

        # 2) Edge spurs per quarter ----------------------------------------
        # Represent a spur as (x, y, dir) placed on a border-adjacent cell
        edge_spurs: list[Tuple[int, int, int]] = []

        def quarter_edge_candidates(label: str) -> list[Tuple[int, int, int]]:
            xs, ys = quarters[label]
            cands: list[Tuple[int, int, int]] = []
            if label == "NW":
                # North edge segment for NW half
                for x in xs:
                    y = 0
                    if x in (mid1, mid2):
                        continue  # avoid exact midline
                    cands.append((x, y, EAST))
                # West edge segment for NW half
                for y in ys:
                    x = 0
                    if y in (mid1, mid2):
                        continue
                    cands.append((x, y, SOUTH))
            elif label == "NE":
                for x in xs:
                    y = 0
                    if x in (mid1, mid2):
                        continue
                    cands.append((x, y, WEST))
                for y in ys:
                    x = size - 1
                    if y in (mid1, mid2):
                        continue
                    cands.append((x, y, SOUTH))
            elif label == "SW":
                for x in xs:
                    y = size - 1
                    if x in (mid1, mid2):
                        continue
                    cands.append((x, y, EAST))
                for y in ys:
                    x = 0
                    if y in (mid1, mid2):
                        continue
                    cands.append((x, y, NORTH))
            elif label == "SE":
                for x in xs:
                    y = size - 1
                    if x in (mid1, mid2):
                        continue
                    cands.append((x, y, WEST))
                for y in ys:
                    x = size - 1
                    if y in (mid1, mid2):
                        continue
                    cands.append((x, y, NORTH))
            # Avoid corners duplication by keeping exactly as generated per quarter
            return cands

        # Place exactly one spur on each edge per quarter
        placed_spur_cells: Set[Tuple[int, int]] = set()
        placed_L_centers: Set[Tuple[int, int]] = set()
        for label in ("NW", "NE", "SW", "SE"):
            cands = quarter_edge_candidates(label)

            # Split candidates into the two edges for this quarter
            if label == "NW":
                edge_a = [(x, y, d) for (x, y, d) in cands if y == 0 and d == EAST]  # North edge
                edge_b = [(x, y, d) for (x, y, d) in cands if x == 0 and d == SOUTH]  # West edge
            elif label == "NE":
                edge_a = [(x, y, d) for (x, y, d) in cands if y == 0 and d == WEST]          # North edge
                edge_b = [(x, y, d) for (x, y, d) in cands if x == size - 1 and d == SOUTH]   # East edge
            elif label == "SW":
                edge_a = [(x, y, d) for (x, y, d) in cands if y == size - 1 and d == EAST]   # South edge
                edge_b = [(x, y, d) for (x, y, d) in cands if x == 0 and d == NORTH]          # West edge
            else:  # "SE"
                edge_a = [(x, y, d) for (x, y, d) in cands if y == size - 1 and d == WEST]   # South edge
                edge_b = [(x, y, d) for (x, y, d) in cands if x == size - 1 and d == NORTH]   # East edge

            rng.shuffle(edge_a)
            rng.shuffle(edge_b)

            # Find a pair with distinct cells and not previously used
            chosen_a = chosen_b = None
            found_pair = False
            for xa, ya, da in edge_a:
                if (xa, ya) in placed_spur_cells:
                    continue
                for xb, yb, db in edge_b:
                    if (xb, yb) in placed_spur_cells:
                        continue
                    if xa == xb and ya == yb:
                        continue
                    chosen_a = (xa, ya, da)
                    chosen_b = (xb, yb, db)
                    found_pair = True
                    break
                if found_pair:
                    break

            if not found_pair:
                # As a fallback, try the reverse pairing order (in case the above shuffles were unlucky)
                for xb, yb, db in edge_b:
                    if (xb, yb) in placed_spur_cells:
                        continue
                    for xa, ya, da in edge_a:
                        if (xa, ya) in placed_spur_cells:
                            continue
                        if xa == xb and ya == yb:
                            continue
                        chosen_a = (xa, ya, da)
                        chosen_b = (xb, yb, db)
                        found_pair = True
                        break
                    if found_pair:
                        break

            if not found_pair:
                # If still not found, skip placing spurs for this quarter (extremely unlikely)
                continue

            # Place the two spurs
            for (x, y, d) in (chosen_a, chosen_b):
                board.add_wall(x, y, d)
                edge_spurs.append((x, y, d))
                placed_spur_cells.add((x, y))

        # 3) L-shaped wall pairs per quarter --------------------------------
        # Orientations (pair of directions applied to the same cell):
        # 0: NORTH+EAST, 1: EAST+SOUTH, 2: SOUTH+WEST, 3: WEST+NORTH
        L_ORIENTS = [
            (NORTH, EAST),  # NE
            (EAST, SOUTH),  # SE
            (SOUTH, WEST),  # SW
            (WEST, NORTH),  # NW
        ]

        def orientation_is_internal(x: int, y: int, orient_idx: int) -> bool:
            d1, d2 = L_ORIENTS[orient_idx]
            # Ensure no wall uses the outer board edge
            if d1 == NORTH and y == 0:
                return False
            if d1 == SOUTH and y == size - 1:
                return False
            if d1 == WEST and x == 0:
                return False
            if d1 == EAST and x == size - 1:
                return False
            if d2 == NORTH and y == 0:
                return False
            if d2 == SOUTH and y == size - 1:
                return False
            if d2 == WEST and x == 0:
                return False
            if d2 == EAST and x == size - 1:
                return False
            return True

        # Compute Chebyshev-based forbidden cells around anchors that have walls
        def chebyshev_forbidden() -> Set[Tuple[int, int]]:
            anchors: Set[Tuple[int, int]] = set()
            # Any cell with at least one wall counts as an anchor
            h, w = board.walls.shape
            for yy in range(h):
                for xx in range(w):
                    if board.walls[yy, xx] != 0:
                        anchors.add((xx, yy))
            forbidden: Set[Tuple[int, int]] = set()
            for ax, ay in anchors:
                forbidden.add((ax, ay))
                for nx, ny in chebyshev_neighbors(ax, ay):
                    forbidden.add((nx, ny))
            return forbidden

        def can_place_L(x: int, y: int, orient_idx: int) -> bool:
            if not orientation_is_internal(x, y, orient_idx):
                return False
            if (x, y) in chebyshev_forbidden():
                return False
            return True

        def place_L(x: int, y: int, orient_idx: int) -> None:
            d1, d2 = L_ORIENTS[orient_idx]
            board.add_wall(x, y, d1)
            board.add_wall(x, y, d2)
            placed_L_centers.add((x, y))

        # Orientation distribution helper per quarter
        def orientation_sequence(total: int) -> List[int]:
            base = total // 4
            rem = total % 4
            counts = [base] * 4
            # Distribute remainders deterministically by shuffling orientation order
            order = [0, 1, 2, 3]
            rng.shuffle(order)
            for i in range(rem):
                counts[order[i]] += 1
            seq: List[int] = []
            # Round-robin to avoid clustering
            # Build per-orientation queues
            queues: List[List[int]] = [[i] * counts[i] for i in range(4)]
            # Interleave
            while any(queues):
                for i in order:
                    if queues[i]:
                        seq.append(queues[i].pop())
            return seq

        # For each quarter, attempt to place up to M L-pairs
        for label in ("NW", "NE", "SW", "SE"):
            xs, ys = quarters[label]
            seq = orientation_sequence(M)

            # For efficiency, precompute candidate lists per orientation filtered by quarter and static constraints
            candidates_by_orient: list[list[Tuple[int, int]]] = [[] for _ in range(4)]
            for y in ys:
                for x in xs:
                    # Skip central forbidden area quickly
                    if (x, y) in central_forbidden:
                        continue
                    for oi in range(4):
                        if orientation_is_internal(x, y, oi):
                            candidates_by_orient[oi].append((x, y))
            # Shuffle candidates for each orientation with the rng for reproducibility
            for lst in candidates_by_orient:
                rng.shuffle(lst)

            # Greedy placement following the interleaved orientation sequence
            for oi in seq:
                placed = False
                cand_list = candidates_by_orient[oi]
                # Try candidates until one fits
                while cand_list:
                    cx, cy = cand_list.pop()
                    if can_place_L(cx, cy, oi):
                        place_L(cx, cy, oi)
                        placed = True
                        break
                if not placed:
                    # Could not place this orientation instance; continue
                    continue

        return board 