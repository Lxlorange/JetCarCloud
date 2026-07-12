from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

import numpy as np

from app.schemas import AlgorithmInfo


class LocalAlgorithm(Protocol):
    def run(
        self,
        *,
        image: np.ndarray,
        spec: AlgorithmInfo,
        car_id: str,
        stream_id: str,
        parameters: dict,
    ) -> tuple[dict, np.ndarray | None]:
        ...


@dataclass
class LocalAlgorithmRunner:
    project_root: Path

    def __post_init__(self) -> None:
        self._instances: dict[str, LocalAlgorithm] = {}
        self._locks: dict[str, Lock] = {}

    def run(
        self,
        *,
        algorithm_id: str,
        spec: AlgorithmInfo,
        image: np.ndarray,
        car_id: str,
        stream_id: str,
        parameters: dict | None = None,
    ) -> tuple[dict, np.ndarray | None]:
        instance = self._instances.get(algorithm_id)
        if instance is None:
            instance = self._create_algorithm(algorithm_id, spec)
            self._instances[algorithm_id] = instance
            self._locks[algorithm_id] = Lock()
        with self._locks[algorithm_id]:
            return instance.run(
                image=image,
                spec=spec,
                car_id=car_id,
                stream_id=stream_id,
                parameters=parameters or {},
            )

    def _create_algorithm(self, algorithm_id: str, spec: AlgorithmInfo) -> LocalAlgorithm:
        task = str(spec.metadata.get("task") or algorithm_id)
        if task in {"manhole_detection", "yolov5-manhole-detect"}:
            from app.algorithms.local.yolov5_manhole import Yolov5ManholeAlgorithm

            return Yolov5ManholeAlgorithm(project_root=self.project_root, spec=spec)
        if task in {"road_damage_detection", "yolov8-road-damage"}:
            from app.algorithms.local.yolov8_road_damage import Yolov8RoadDamageAlgorithm

            return Yolov8RoadDamageAlgorithm(project_root=self.project_root, spec=spec)
        if task in {"similarity", "yolov5-similarity"}:
            from app.algorithms.local.similarity import SimilarityAlgorithm

            return SimilarityAlgorithm(project_root=self.project_root, spec=spec)
        raise KeyError(f"unsupported local algorithm task: {task}")


def resolve_project_path(project_root: Path, value: str | None, default: str) -> Path:
    raw = value or default
    path = Path(raw)
    if path.is_absolute():
        return path
    return project_root / path
