from __future__ import annotations

import json
from pathlib import Path

from app.schemas import AlgorithmInfo


class AlgorithmCatalog:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._items = self._load(self._path)

    def list(self) -> list[AlgorithmInfo]:
        return list(self._items.values())

    def get(self, algorithm_id: str) -> AlgorithmInfo | None:
        return self._items.get(algorithm_id)

    def require(self, algorithm_id: str) -> AlgorithmInfo:
        item = self.get(algorithm_id)
        if item is None:
            raise KeyError(f"algorithm not found: {algorithm_id}")
        if not item.enabled:
            raise KeyError(f"algorithm disabled: {algorithm_id}")
        return item

    def reload(self) -> list[AlgorithmInfo]:
        self._items = self._load(self._path)
        return self.list()

    def _load(self, path: Path) -> dict[str, AlgorithmInfo]:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        algorithms = raw.get("algorithms", raw)
        items: dict[str, AlgorithmInfo] = {}
        for algorithm_id, spec in algorithms.items():
            items[algorithm_id] = AlgorithmInfo(algorithm_id=algorithm_id, **spec)
        return items
