"""
Utility helpers for saving / loading agents, game states, and training metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pickle


class CheckpointManager:
    """
    Very thin wrapper around Pickle / JSON until we need something fancier.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Agents ---------------- #

    def save_agent(self, agent, name: str) -> Path:
        path = self.root_dir / f"{name}.pkl"
        with path.open("wb") as fh:
            pickle.dump(agent, fh)
        return path

    def load_agent(self, name: str):
        path = self.root_dir / f"{name}.pkl"
        with path.open("rb") as fh:
            return pickle.load(fh)

    # ---------------- GameState ---------------- #

    def save_state(self, state, name: str) -> Path:
        path = self.root_dir / f"{name}.pkl"
        with path.open("wb") as fh:
            pickle.dump(state, fh)
        return path

    def load_state(self, name: str):
        path = self.root_dir / f"{name}.pkl"
        with path.open("rb") as fh:
            return pickle.load(fh)

    # ---------------- Misc ---------------- #

    def save_json(self, data: Dict[str, Any], name: str) -> Path:
        path = self.root_dir / f"{name}.json"
        with path.open("w") as fh:
            json.dump(data, fh, indent=2)
        return path

    def load_json(self, name: str) -> Dict[str, Any]:
        path = self.root_dir / f"{name}.json"
        with path.open() as fh:
            return json.load(fh) 