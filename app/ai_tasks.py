from __future__ import annotations

import time

import numpy as np

from app.inference.detector import Detector
from app.schemas import AiTaskResult, AiTaskSpec


class AiTaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, AiTaskSpec] = {}

    def register(self, spec: AiTaskSpec) -> AiTaskSpec:
        self._tasks[spec.task_id] = spec
        return spec

    def get(self, task_id: str) -> AiTaskSpec | None:
        return self._tasks.get(task_id)

    def list(self) -> list[AiTaskSpec]:
        return list(self._tasks.values())


def run_yolo_task(
    *,
    task_id: str,
    detector: Detector,
    image: np.ndarray,
    car_id: str,
    stream_id: str,
    sensors: dict | None = None,
    metadata: dict | None = None,
) -> AiTaskResult:
    started = time.perf_counter()
    detections = detector.detect(image, sensors or {})
    latency_ms = (time.perf_counter() - started) * 1000.0
    return AiTaskResult(
        task_id=task_id,
        car_id=car_id,
        stream_id=stream_id,
        latency_ms=round(latency_ms, 3),
        detections=detections,
        metadata=metadata or {},
    )
