from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SimilarityOutput:
    similarity: float
    matched: bool
    threshold: float
    method: str
    yolo_summary: dict[str, Any]
    latency_ms: float


def unit_normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector
    return vector / norm


def extract_feature_vector(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.size == 0:
        return np.zeros(1, dtype=np.float32)

    resized = cv2.resize(image_bgr, (160, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    color_feature = unit_normalize(np.concatenate([hist_h, hist_s, hist_v]))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    if hasattr(cv2, "HOGDescriptor"):
        hog = cv2.HOGDescriptor(
            _winSize=(160, 160),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        shape_feature = unit_normalize(hog.compute(gray).reshape(-1))
    else:
        # Some minimal OpenCV builds do not expose HOGDescriptor. Keep the
        # pipeline available with a compact grayscale shape descriptor.
        shape_feature = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).reshape(-1)
        shape_feature = unit_normalize(shape_feature)

    edges = cv2.Canny(gray, 80, 160)
    edge_feature = cv2.resize(edges, (32, 32), interpolation=cv2.INTER_AREA).reshape(-1)
    edge_feature = unit_normalize(edge_feature)

    return unit_normalize(np.concatenate([color_feature, shape_feature * 1.5, edge_feature]))


def cosine_similarity(feature_a: np.ndarray, feature_b: np.ndarray) -> float:
    a = unit_normalize(feature_a)
    b = unit_normalize(feature_b)
    if a.shape != b.shape:
        return 0.0
    score = float(np.dot(a, b))
    return max(0.0, min(1.0, score))


def detection_summary(detector: Any, image_bgr: np.ndarray) -> dict[str, Any]:
    try:
        detections = detector.detect(image_bgr, {})
    except Exception as exc:
        return {"available": False, "error": str(exc), "labels": [], "count": 0}

    labels: list[str] = []
    confidences: list[float] = []
    for item in detections:
        label = getattr(item, "label", None)
        confidence = getattr(item, "confidence", None)
        if label is not None:
            labels.append(str(label))
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    return {
        "available": detector.__class__.__name__ != "NullDetector",
        "count": len(detections),
        "labels": labels,
        "max_confidence": max(confidences) if confidences else None,
    }


def compare_images(
    reference_bgr: np.ndarray,
    query_bgr: np.ndarray,
    *,
    detector: Any,
    threshold: float = 0.45,
) -> SimilarityOutput:
    started = time.perf_counter()

    ref_feature = extract_feature_vector(reference_bgr)
    query_feature = extract_feature_vector(query_bgr)
    image_similarity = cosine_similarity(ref_feature, query_feature)

    ref_yolo = detection_summary(detector, reference_bgr)
    query_yolo = detection_summary(detector, query_bgr)
    shared_labels = sorted(set(ref_yolo["labels"]) & set(query_yolo["labels"]))
    yolo_bonus = 0.08 if shared_labels else 0.0

    similarity = max(0.0, min(1.0, image_similarity + yolo_bonus))
    latency_ms = (time.perf_counter() - started) * 1000.0
    return SimilarityOutput(
        similarity=round(similarity, 4),
        matched=similarity >= threshold,
        threshold=threshold,
        method="opencv_feature_cosine+yolo_label_bonus",
        yolo_summary={
            "reference": ref_yolo,
            "query": query_yolo,
            "shared_labels": shared_labels,
            "label_bonus": yolo_bonus,
        },
        latency_ms=round(latency_ms, 3),
    )
