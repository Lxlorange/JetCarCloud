from __future__ import annotations

import cv2
import numpy as np

from app.schemas import Detection


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    *,
    color: tuple[int, int, int],
    prefix: str = "",
) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection.bbox]
        label = f"{prefix}{detection.label} {detection.confidence:.2f}".strip()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_label(annotated, label, x1, y1, color)
    return annotated


def draw_detection_groups(
    image: np.ndarray,
    groups: list[tuple[str, list[Detection], tuple[int, int, int]]],
) -> np.ndarray:
    annotated = image.copy()
    for prefix, detections, color in groups:
        annotated = draw_detections(annotated, detections, color=color, prefix=prefix)
    return annotated


def _draw_label(image: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y - text_height - baseline - 4)
    left = max(0, x)
    cv2.rectangle(
        image,
        (left, top),
        (left + text_width + 6, top + text_height + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (left + 3, top + text_height + 1),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
