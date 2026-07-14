from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.algorithms.local.runner import resolve_project_path
from app.schemas import AlgorithmInfo


class Yolov8RoadDamageAlgorithm:
    def __init__(self, *, project_root: Path, spec: AlgorithmInfo) -> None:
        from ultralytics import YOLO

        self._project_root = project_root
        model_path = resolve_project_path(
            project_root,
            str(spec.metadata.get("model_path") or ""),
            "models/yolov8-road-damage/YOLOv8_Small_2nd_Model.pt",
        )
        if not model_path.exists():
            raise FileNotFoundError(f"YOLOv8 model file not found: {model_path}")
        self._model_path = model_path
        self._model = YOLO(str(model_path))

    def run(
        self,
        *,
        image: np.ndarray,
        spec: AlgorithmInfo,
        car_id: str,
        stream_id: str,
        parameters: dict,
    ) -> tuple[dict, np.ndarray | None]:
        imgsz = int(parameters.get("imgsz", spec.metadata.get("imgsz", 640)))
        conf = float(parameters.get("conf", spec.metadata.get("conf", 0.20)))
        iou = float(parameters.get("iou", spec.metadata.get("iou", 0.45)))
        max_det = int(parameters.get("max_det", spec.metadata.get("max_det", 50)))
        device = str(parameters.get("device", spec.metadata.get("device", "cpu")))

        kwargs = {
            "source": image,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "verbose": False,
        }
        if device.lower() != "auto":
            kwargs["device"] = device

        results = self._model.predict(**kwargs)
        if not results:
            raise RuntimeError("YOLOv8 model returned no prediction result")

        result = results[0]
        annotated = result.plot()
        detections = _collect_yolov8_detections(result, self._model.names)
        return (
            {
                "algorithm_id": spec.algorithm_id,
                "car_id": car_id,
                "stream_id": stream_id,
                "task": "road_damage_detection",
                "model_path": str(self._model_path),
                "detection_count": len(detections),
                "class_summary": dict(Counter(item["class_name"] for item in detections)),
                "detections": detections,
                "config": {"imgsz": imgsz, "conf": conf, "iou": iou, "max_det": max_det, "device": device},
            },
            annotated,
        )


def _collect_yolov8_detections(result, names: dict[int, str]) -> list[dict]:
    detections: list[dict] = []
    if result.boxes is None:
        return detections

    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        detections.append(
            {
                "id": index,
                "class_id": class_id,
                "class_name": str(names.get(class_id, class_id)),
                "confidence": round(confidence, 6),
                "bbox_xyxy": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
            }
        )
    return detections
