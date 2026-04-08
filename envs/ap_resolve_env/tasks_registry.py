"""
TaskRegistry — loads and indexes fixture files from fixtures/.

Provides get(), random(), and list_ids() methods.
"""

from __future__ import annotations

import json
import random as _random
from pathlib import Path
from typing import List

from .models import CaseFixture

_TASKS_DIR = Path(__file__).parent / "tasks"

TASK_IDS: List[str] = [
    "easy_straight_through",
    "medium_mismatch",
    "hard_duplicate_partial",
]


class TaskRegistry:
    """Loads and indexes AP-Resolve fixture files."""

    def __init__(self, tasks_dir: Path | None = None) -> None:
        self._dir = tasks_dir or _TASKS_DIR
        self._cache: dict[str, CaseFixture] = {}
        self._load_all()

    def _load_all(self) -> None:
        for task_id in TASK_IDS:
            path = self._dir / f"{task_id}.json"
            if not path.exists():
                raise ValueError(
                    f"Fixture file not found: {path}. "
                    f"Expected fixture for task_id='{task_id}'."
                )
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed fixture file {path}: {exc}"
                ) from exc
            self._cache[task_id] = CaseFixture(**data)

    def get(self, task_id: str) -> CaseFixture:
        """Return the fixture for the given task_id.

        Raises ValueError if task_id is unknown.
        """
        if task_id not in self._cache:
            raise ValueError(
                f"Unknown task_id: '{task_id}'. "
                f"Valid IDs: {self.list_ids()}"
            )
        return self._cache[task_id]

    def random(self) -> CaseFixture:
        """Return a uniformly random fixture."""
        task_id = _random.choice(TASK_IDS)
        return self._cache[task_id]

    def list_ids(self) -> List[str]:
        """Return the list of all registered task IDs."""
        return list(TASK_IDS)
