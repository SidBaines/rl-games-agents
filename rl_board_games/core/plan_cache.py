from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


class PlanDifficultyCache:
    """
    Persistent cache of A* plan features per (board_size, num_robots, seed).

    Stores per-seed features to enable fast classification into curriculum levels
    without re-running A* solves every time.

    File format (JSON):
    { "seed": {"total_moves": int, "robots_moved": int}, ... }
    """

    def __init__(self, lookup_dir: str | Path = "plan_lookup") -> None:
        self.lookup_dir = Path(lookup_dir)
        self.lookup_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[int, int], Dict[int, Dict[str, int]]] = {}

    def _get_filename(self, board_size: int, num_robots: int) -> Path:
        return self.lookup_dir / f"plan_lookup_{board_size}x{board_size}_{num_robots}robots.json"

    def load_table(self, board_size: int, num_robots: int) -> Dict[int, Dict[str, int]]:
        key = (board_size, num_robots)
        if key in self._cache:
            return self._cache[key]
        path = self._get_filename(board_size, num_robots)
        if path.exists():
            with path.open("r") as f:
                data = json.load(f)
                table = {int(k): {"total_moves": int(v.get("total_moves", 0)), "robots_moved": int(v.get("robots_moved", 0))} for k, v in data.items()}
                self._cache[key] = table
                return table
        return {}

    def save_table(self, board_size: int, num_robots: int, table: Dict[int, Dict[str, int]]) -> None:
        path = self._get_filename(board_size, num_robots)
        with path.open("w") as f:
            json.dump({str(k): v for k, v in table.items()}, f, indent=2)
        self._cache[(board_size, num_robots)] = table

    def get(self, seed: int, board_size: int, num_robots: int) -> Optional[Dict[str, int]]:
        table = self.load_table(board_size, num_robots)
        return table.get(seed)

    def add(self, seed: int, board_size: int, num_robots: int, total_moves: int, robots_moved: int) -> None:
        table = self.load_table(board_size, num_robots)
        table[seed] = {"total_moves": int(total_moves), "robots_moved": int(robots_moved)}
        self.save_table(board_size, num_robots, table)

    def get_matching_seeds(
        self,
        board_size: int,
        num_robots: int,
        predicate,
    ) -> List[int]:
        """Return seeds from cache that satisfy the given predicate(features)->bool."""
        table = self.load_table(board_size, num_robots)
        return [seed for seed, feats in table.items() if predicate(feats)]

    def entries(self, board_size: int, num_robots: int) -> Iterable[Tuple[int, Dict[str, int]]]:
        table = self.load_table(board_size, num_robots)
        return table.items() 