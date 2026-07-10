from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.inference.fusion import estimate_distance_m
from app.schemas import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, sensors: dict[str, Any]) -> list[Detection]:
        raise NotImplementedError


class NullDetector(Detector):
    def detect(self, image: np.ndarray, sensors: dict[str, Any]) -> list[Detection]:
        return []


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._device = device
        self._confidence = confidence
        self._image_size = image_size

    def detect(self, image: np.ndarray, sensors: dict[str, Any]) -> list[Detection]:
        results = self._model.predict(
            source=image,
            imgsz=self._image_size,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        if not results:
            return []

        names = results[0].names
        detections: list[Detection] = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(cls_id, cls_id))
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=[float(v) for v in xyxy],
                    distance_m=estimate_distance_m(xyxy, sensors),
                )
            )
        return detections


def build_detector(
    model_path: str,
    *,
    device: str = "cpu",
    confidence: float = 0.25,
    image_size: int = 640,
) -> Detector:
    if not model_path:
        return NullDetector()

    path = Path(model_path)
    if not path.exists():
        return NullDetector()

    return YoloDetector(
        str(path),
        device=device,
        confidence=confidence,
        image_size=image_size,
    )

