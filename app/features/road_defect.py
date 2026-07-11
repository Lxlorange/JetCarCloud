from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.features.annotate import draw_detections
from app.inference.detector import NullDetector, build_detector
from app.schemas import RoadDefectDetectionResult
from app.video import encode_jpeg_payload


@dataclass(frozen=True)
class RoadDefectSettings:
    model_path: str
    backend: str
    device: str
    confidence: float
    image_size: int
    positive_labels: set[str]


class RoadDefectProvider:
    name = "local"

    def __init__(self, settings: RoadDefectSettings) -> None:
        self._positive_labels = settings.positive_labels
        self._error = ""
        if not settings.model_path:
            self.name = "none"
            self._detector = NullDetector("ROAD_DEFECT_MODEL_PATH is empty")
            self._error = self._detector.reason
            return
        if not Path(settings.model_path).exists():
            self.name = "none"
            self._detector = NullDetector(f"ROAD_DEFECT_MODEL_PATH does not exist: {settings.model_path}")
            self._error = self._detector.reason
            return

        self._detector = build_detector(
            settings.model_path,
            backend=settings.backend,
            device=settings.device,
            confidence=settings.confidence,
            image_size=settings.image_size,
        )
        if isinstance(self._detector, NullDetector):
            self.name = "none"
            self._error = self._detector.reason

    def detect(
        self,
        image: np.ndarray,
        *,
        car_id: str,
        stream_id: str,
        metadata: dict | None = None,
        include_image: bool = False,
    ) -> RoadDefectDetectionResult:
        if isinstance(self._detector, NullDetector):
            return RoadDefectDetectionResult(
                ok=False,
                car_id=car_id,
                stream_id=stream_id,
                provider="none",
                metadata=metadata or {},
                error=self._error,
            )

        started = time.perf_counter()
        detections = self._detector.detect(image, {})
        filtered = _filter_detections(detections, self._positive_labels)
        latency_ms = (time.perf_counter() - started) * 1000.0
        annotated = None
        if include_image:
            annotated = encode_jpeg_payload(draw_detections(image, filtered, color=(0, 0, 255), prefix="defect:"))
        return RoadDefectDetectionResult(
            car_id=car_id,
            stream_id=stream_id,
            provider="local",
            found=bool(filtered),
            count=len(filtered),
            latency_ms=round(latency_ms, 3),
            detections=filtered,
            metadata=metadata or {},
            annotated_image=annotated,
        )


def build_road_defect_provider(settings: RoadDefectSettings) -> RoadDefectProvider:
    return RoadDefectProvider(settings)


def _filter_detections(detections, positive_labels: set[str]):
    if not positive_labels:
        return detections
    return [detection for detection in detections if detection.label.lower() in positive_labels]
