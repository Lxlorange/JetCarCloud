from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from app.features.annotate import draw_detections
from app.inference.detector import Detector, NullDetector, build_detector
from app.schemas import Detection, ManholeDetectionResult
from app.video import encode_jpeg_payload


@dataclass(frozen=True)
class ManholeSettings:
    provider: str
    model_path: str
    backend: str
    yolov5_repo_path: str
    device: str
    confidence: float
    image_size: int
    positive_labels: set[str]
    roboflow_api_key: str
    roboflow_model_id: str
    roboflow_model_version: str
    roboflow_api_url: str


class ManholeProvider(ABC):
    name: str

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        *,
        car_id: str,
        stream_id: str,
        metadata: dict | None = None,
        include_image: bool = False,
    ) -> ManholeDetectionResult:
        raise NotImplementedError


class LocalYoloManholeProvider(ManholeProvider):
    name = "local"

    def __init__(self, detector: Detector, positive_labels: set[str]) -> None:
        self._detector = detector
        self._positive_labels = positive_labels

    def detect(
        self,
        image: np.ndarray,
        *,
        car_id: str,
        stream_id: str,
        metadata: dict | None = None,
        include_image: bool = False,
    ) -> ManholeDetectionResult:
        started = time.perf_counter()
        detections = self._detector.detect(image, {})
        filtered = _filter_manhole_detections(detections, self._positive_labels)
        latency_ms = (time.perf_counter() - started) * 1000.0
        annotated = None
        if include_image:
            annotated = encode_jpeg_payload(draw_detections(image, filtered, color=(0, 180, 255), prefix="manhole:"))
        return ManholeDetectionResult(
            car_id=car_id,
            stream_id=stream_id,
            provider=self.name,
            found=bool(filtered),
            count=len(filtered),
            latency_ms=round(latency_ms, 3),
            detections=filtered,
            metadata=metadata or {},
            error=getattr(self._detector, "reason", ""),
            annotated_image=annotated,
        )


class RoboflowManholeProvider(ManholeProvider):
    name = "roboflow"

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        model_version: str,
        api_url: str = "",
        confidence: float = 0.25,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._model_version = model_version
        self._api_url = api_url
        self._confidence = confidence

    def detect(
        self,
        image: np.ndarray,
        *,
        car_id: str,
        stream_id: str,
        metadata: dict | None = None,
        include_image: bool = False,
    ) -> ManholeDetectionResult:
        started = time.perf_counter()
        payload = encode_jpeg_payload(image)
        response = requests.post(
            self._prediction_url(),
            params={"api_key": self._api_key},
            data=payload.data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=(5, 30),
        )
        response.raise_for_status()
        data = response.json()
        detections = [
            _roboflow_prediction_to_detection(prediction)
            for prediction in data.get("predictions", [])
            if float(prediction.get("confidence", 0.0)) >= self._confidence
        ]
        latency_ms = (time.perf_counter() - started) * 1000.0
        annotated = None
        if include_image:
            annotated = encode_jpeg_payload(draw_detections(image, detections, color=(0, 180, 255), prefix="manhole:"))
        return ManholeDetectionResult(
            car_id=car_id,
            stream_id=stream_id,
            provider=self.name,
            found=bool(detections),
            count=len(detections),
            latency_ms=round(latency_ms, 3),
            detections=detections,
            metadata={
                **(metadata or {}),
                "roboflow": {
                    "model_id": self._model_id,
                    "model_version": self._model_version,
                    "image": data.get("image", {}),
                },
            },
            annotated_image=annotated,
        )

    def _prediction_url(self) -> str:
        if self._api_url:
            return self._api_url.rstrip("/")
        return f"https://detect.roboflow.com/{self._model_id}/{self._model_version}"


class UnavailableManholeProvider(ManholeProvider):
    name = "none"

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def detect(
        self,
        image: np.ndarray,
        *,
        car_id: str,
        stream_id: str,
        metadata: dict | None = None,
        include_image: bool = False,
    ) -> ManholeDetectionResult:
        return ManholeDetectionResult(
            ok=False,
            car_id=car_id,
            stream_id=stream_id,
            provider=self.name,
            metadata=metadata or {},
            error=self._reason,
        )


def build_manhole_provider(settings: ManholeSettings) -> ManholeProvider:
    provider = settings.provider.lower().strip()
    if provider == "roboflow":
        if not settings.roboflow_api_key:
            return UnavailableManholeProvider("ROBOFLOW_API_KEY is empty")
        return RoboflowManholeProvider(
            api_key=settings.roboflow_api_key,
            model_id=settings.roboflow_model_id,
            model_version=settings.roboflow_model_version,
            api_url=settings.roboflow_api_url,
            confidence=settings.confidence,
        )

    if provider not in {"local", "yolov5", "auto"}:
        return UnavailableManholeProvider(f"unsupported MANHOLE_PROVIDER: {settings.provider}")

    if not settings.model_path:
        return UnavailableManholeProvider("MANHOLE_MODEL_PATH is empty")
    if not Path(settings.model_path).exists():
        return UnavailableManholeProvider(f"MANHOLE_MODEL_PATH does not exist: {settings.model_path}")

    detector = build_detector(
        settings.model_path,
        backend=settings.backend,
        yolov5_repo_path=settings.yolov5_repo_path,
        device=settings.device,
        confidence=settings.confidence,
        image_size=settings.image_size,
    )
    if isinstance(detector, NullDetector):
        return UnavailableManholeProvider(detector.reason)
    return LocalYoloManholeProvider(detector, settings.positive_labels)


def _filter_manhole_detections(detections: list[Detection], positive_labels: set[str]) -> list[Detection]:
    if not positive_labels:
        return detections
    return [detection for detection in detections if detection.label.lower() in positive_labels]


def _roboflow_prediction_to_detection(prediction: dict) -> Detection:
    x = float(prediction.get("x", 0.0))
    y = float(prediction.get("y", 0.0))
    width = float(prediction.get("width", 0.0))
    height = float(prediction.get("height", 0.0))
    return Detection(
        label=str(prediction.get("class", "manhole")),
        confidence=float(prediction.get("confidence", 0.0)),
        bbox=[
            x - width / 2.0,
            y - height / 2.0,
            x + width / 2.0,
            y + height / 2.0,
        ],
    )
