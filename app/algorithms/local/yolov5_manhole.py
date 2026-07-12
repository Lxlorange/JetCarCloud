from __future__ import annotations

import os
import sys
from pathlib import Path

if os.name == "nt":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import torch

from app.algorithms.local.runner import resolve_project_path
from app.schemas import AlgorithmInfo


class Yolov5ManholeAlgorithm:
    def __init__(self, *, project_root: Path, spec: AlgorithmInfo) -> None:
        self._project_root = project_root
        self._source_dir = resolve_project_path(
            project_root,
            str(spec.metadata.get("source_dir") or ""),
            "models/yolov5-manhole/src",
        ).resolve()
        if not self._source_dir.exists():
            raise FileNotFoundError(f"YOLOv5 source directory not found: {self._source_dir}")
        if str(self._source_dir) not in sys.path:
            sys.path.insert(0, str(self._source_dir))

        from models.experimental import attempt_load
        from utils.general import check_img_size

        self._letterbox, self._non_max_suppression, self._scale_boxes = _load_yolov5_utils()

        model_path = resolve_project_path(
            project_root,
            str(spec.metadata.get("model_path") or ""),
            "models/yolov5-manhole/Manhole_model.pt",
        )
        if not model_path.exists():
            raise FileNotFoundError(f"YOLOv5 manhole model file not found: {model_path}")
        self._model_path = model_path
        self._device = _select_device(str(spec.metadata.get("device", "")))
        self._model = attempt_load(str(model_path), device=self._device, inplace=True, fuse=True)
        self._model.eval()
        self._stride = int(self._model.stride.max()) if hasattr(self._model, "stride") else 32
        self._names = getattr(self._model, "names", {0: "manhole"})
        self._default_img_size = check_img_size(int(spec.metadata.get("img_size", 640)), s=self._stride)

    def run(
        self,
        *,
        image: np.ndarray,
        spec: AlgorithmInfo,
        car_id: str,
        stream_id: str,
        parameters: dict,
    ) -> tuple[dict, np.ndarray | None]:
        from utils.general import check_img_size

        img_size = check_img_size(int(parameters.get("img_size", self._default_img_size)), s=self._stride)
        conf_threshold = float(parameters.get("conf_threshold", spec.metadata.get("conf_threshold", 0.25)))
        iou_threshold = float(parameters.get("iou_threshold", spec.metadata.get("iou_threshold", 0.45)))
        max_det = int(parameters.get("max_det", spec.metadata.get("max_det", 1000)))

        tensor = self._preprocess(image, img_size)
        with torch.no_grad():
            prediction = self._model(tensor, augment=False)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            prediction = self._non_max_suppression(
                prediction,
                conf_threshold,
                iou_threshold,
                max_det=max_det,
            )

        annotated = image.copy()
        detections: list[dict] = []
        for det in prediction:
            if len(det):
                det[:, :4] = self._scale_boxes(tensor.shape[2:], det[:, :4], image.shape).round()
                for index, (*xyxy, conf, cls) in enumerate(det.tolist()):
                    class_id = int(cls)
                    label = self._names[class_id] if isinstance(self._names, dict) else self._names[class_id]
                    confidence = float(conf)
                    box = [int(v) for v in xyxy]
                    _draw_detection(annotated, box, label, confidence)
                    detections.append(
                        {
                            "id": index,
                            "class_id": class_id,
                            "class_name": str(label),
                            "confidence": round(confidence, 6),
                            "bbox_xyxy": box,
                        }
                    )

        return (
            {
                "algorithm_id": spec.algorithm_id,
                "car_id": car_id,
                "stream_id": stream_id,
                "task": "manhole_detection",
                "model_path": str(self._model_path),
                "source_dir": str(self._source_dir),
                "device": str(self._device),
                "img_size": img_size,
                "conf_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "detection_count": len(detections),
                "detections": detections,
            },
            annotated,
        )

    def _preprocess(self, frame: np.ndarray, img_size: int):
        image = self._letterbox(frame, img_size, stride=self._stride, auto=True)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).to(self._device)
        tensor = tensor.float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor


def _load_yolov5_utils():
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    return letterbox, non_max_suppression, scale_boxes


def _select_device(device_value: str):
    if device_value:
        return torch.device(device_value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _draw_detection(image: np.ndarray, xyxy: list[int], label: str, confidence: float) -> None:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
