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
    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def detect(self, image: np.ndarray, sensors: dict[str, Any]) -> list[Detection]:
        return []


class UltralyticsDetector(Detector):
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


class YoloV5Detector(Detector):
    def __init__(
        self,
        model_path: str,
        *,
        repo_path: str,
        device: str = "cpu",
        confidence: float = 0.25,
    ) -> None:
        import torch

        repo = Path(repo_path)
        if not repo.exists() or not (repo / "hubconf.py").exists():
            raise FileNotFoundError(f"YOLOv5 repo with hubconf.py not found: {repo}")

        self._model = torch.hub.load(
            str(repo),
            "custom",
            path=str(Path(model_path)),
            source="local",
            trust_repo=True,
        )
        self._model.to(device)
        self._model.conf = confidence
        self._model.eval()

    def detect(self, image: np.ndarray, sensors: dict[str, Any]) -> list[Detection]:
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._model(image_rgb)
        table = results.pandas().xyxy[0]
        detections: list[Detection] = []
        for _, row in table.iterrows():
            xyxy = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            detections.append(
                Detection(
                    label=str(row["name"]),
                    confidence=float(row["confidence"]),
                    bbox=xyxy,
                    distance_m=estimate_distance_m(xyxy, sensors),
                )
            )
        return detections


def build_detector(
    model_path: str,
    *,
    backend: str = "auto",
    yolov5_repo_path: str = "",
    device: str = "cpu",
    confidence: float = 0.25,
    image_size: int = 640,
) -> Detector:
    if not model_path:
        return NullDetector("YOLO_MODEL_PATH is empty")

    path = Path(model_path)
    if not path.exists():
        return NullDetector(f"YOLO_MODEL_PATH does not exist: {path}")

    try:
        if backend == "yolov5":
            return YoloV5Detector(
                str(path),
                repo_path=yolov5_repo_path,
                device=device,
                confidence=confidence,
            )

        if backend == "ultralytics":
            return UltralyticsDetector(
                str(path),
                device=device,
                confidence=confidence,
                image_size=image_size,
            )

        if yolov5_repo_path:
            return YoloV5Detector(
                str(path),
                repo_path=yolov5_repo_path,
                device=device,
                confidence=confidence,
            )

        return UltralyticsDetector(
            str(path),
            device=device,
            confidence=confidence,
            image_size=image_size,
        )
    except Exception as exc:
        return NullDetector(f"failed to load detector: {exc}")
