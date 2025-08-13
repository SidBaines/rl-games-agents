from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set


class PlanDifficultyCache:
    """
    Persistent cache of A* plan features per (board_size, num_robots, seed).

    Stores per-seed features to enable fast classification into curriculum levels
    without re-running A* solves every time.

    File format (JSON):
    { "seed": {"total_moves": int, "robots_moved": int}, ... }

    Additionally maintains per-combination files to speed up constrained sampling:
    plan_lookup_{WxH}_{R}robots_moves{m}_robots{r}.json -> [seed, seed, ...]
    """

    def __init__(self, lookup_dir: str | Path = "plan_lookup") -> None:
        self.lookup_dir = Path(lookup_dir)
        self.lookup_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[int, int], Dict[int, Dict[str, int]]] = {}

    def _get_index_filename(self, board_size: int, num_robots: int) -> Path:
        return self.lookup_dir / f"plan_lookup_{board_size}x{board_size}_{num_robots}robots.json"

    def _get_combo_filename(self, board_size: int, num_robots: int, total_moves: int, robots_moved: int) -> Path:
        return self.lookup_dir / (
            f"plan_lookup_{board_size}x{board_size}_{num_robots}robots_moves{int(total_moves)}_robots{int(robots_moved)}.json"
        )

    def load_table(self, board_size: int, num_robots: int) -> Dict[int, Dict[str, int]]:
        key = (board_size, num_robots)
        if key in self._cache:
            return self._cache[key]
        path = self._get_index_filename(board_size, num_robots)
        if path.exists():
            with path.open("r") as f:
                data = json.load(f)
                table = {
                    int(k): {
                        "total_moves": int(v.get("total_moves", 0)),
                        "robots_moved": int(v.get("robots_moved", 0)),
                    }
                    for k, v in data.items()
                }
                self._cache[key] = table
                return table
        return {}

    def save_table(self, board_size: int, num_robots: int, table: Dict[int, Dict[str, int]]) -> None:
        path = self._get_index_filename(board_size, num_robots)
        with path.open("w") as f:
            json.dump({str(k): v for k, v in table.items()}, f, indent=2)
        self._cache[(board_size, num_robots)] = table

    def _append_to_combo_file(self, board_size: int, num_robots: int, total_moves: int, robots_moved: int, seed: int) -> None:
        combo_path = self._get_combo_filename(board_size, num_robots, total_moves, robots_moved)
        seeds: List[int] = []
        if combo_path.exists():
            try:
                with combo_path.open("r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        seeds = [int(s) for s in loaded]
            except Exception:
                seeds = []
        if seed not in seeds:
            seeds.append(int(seed))
            with combo_path.open("w") as f:
                json.dump(seeds, f)

    def get(self, seed: int, board_size: int, num_robots: int) -> Optional[Dict[str, int]]:
        table = self.load_table(board_size, num_robots)
        return table.get(seed)

    def add(self, seed: int, board_size: int, num_robots: int, total_moves: int, robots_moved: int) -> None:
        table = self.load_table(board_size, num_robots)
        table[seed] = {"total_moves": int(total_moves), "robots_moved": int(robots_moved)}
        self.save_table(board_size, num_robots, table)
        # Also update per-combination file for faster constrained lookup
        self._append_to_combo_file(board_size, num_robots, total_moves, robots_moved, seed)

    def get_matching_seeds(
        self,
        board_size: int,
        num_robots: int,
        predicate,
    ) -> List[int]:
        """Return seeds from cache that satisfy the given predicate(features)->bool.

        This scans the per-seed index. For faster constrained sampling prefer
        sample_seed_by_constraints which uses per-combination files.
        """
        table = self.load_table(board_size, num_robots)
        return [seed for seed, feats in table.items() if predicate(feats)]

    def entries(self, board_size: int, num_robots: int) -> Iterable[Tuple[int, Dict[str, int]]]:
        table = self.load_table(board_size, num_robots)
        return table.items()

    def sample_seed_by_constraints(
        self,
        board_size: int,
        num_robots: int,
        min_total_moves: int,
        max_total_moves: int,
        max_robots_moved: int,
        rng: Optional[random.Random] = None,
        avoid_seeds: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """
        Sample a seed by first randomly choosing a (total_moves, robots_moved) combination
        within the provided bounds, then selecting a random seed from the corresponding
        per-combination file. Skips empty combinations. Optionally avoids seeds in avoid_seeds.
        """
        rng = rng or random
        avoid: Set[int] = avoid_seeds or set()

        available_combos: List[Tuple[int, int, List[int]]] = []
        for moves in range(int(min_total_moves), int(max_total_moves) + 1):
            for robots in range(0, int(max_robots_moved) + 1):
                combo_path = self._get_combo_filename(board_size, num_robots, moves, robots)
                if not combo_path.exists():
                    continue
                try:
                    with combo_path.open("r") as f:
                        seeds_list = [int(s) for s in json.load(f) if int(s) not in avoid]
                except Exception:
                    seeds_list = []
                if seeds_list:
                    available_combos.append((moves, robots, seeds_list))

        if not available_combos:
            return None

        moves, robots, seeds_list = rng.choice(available_combos)
        return rng.choice(seeds_list) if seeds_list else None 