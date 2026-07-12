from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.algorithms.local.runner import resolve_project_path
from app.schemas import AlgorithmInfo
from app.similarity import extract_feature_vector


class SimilarityAlgorithm:
    def __init__(self, *, project_root: Path, spec: AlgorithmInfo) -> None:
        self._project_root = project_root
        self._default_template_path = resolve_project_path(
            project_root,
            str(spec.metadata.get("template_path") or ""),
            "models/yolov5-similarity/target.jpg",
        )

    def run(
        self,
        *,
        image: np.ndarray,
        spec: AlgorithmInfo,
        car_id: str,
        stream_id: str,
        parameters: dict,
    ) -> tuple[dict, np.ndarray | None]:
        threshold = float(parameters.get("threshold", spec.metadata.get("threshold", 0.45)))
        template_path = resolve_project_path(
            self._project_root,
            str(parameters.get("template_path") or spec.metadata.get("template_path") or ""),
            str(self._default_template_path),
        )
        if not template_path.exists():
            result = {
                "algorithm_id": spec.algorithm_id,
                "car_id": car_id,
                "stream_id": stream_id,
                "task": "similarity",
                "matched": False,
                "similarity": 0.0,
                "threshold": threshold,
                "error": f"template image not found: {template_path}",
            }
            return result, image.copy()

        template = cv2.imread(str(template_path))
        if template is None:
            raise ValueError(f"failed to read template image: {template_path}")

        source_feature = extract_feature_vector(image)
        template_feature = extract_feature_vector(template)
        similarity = _cosine_similarity(source_feature, template_feature)
        matched = similarity >= threshold

        annotated = image.copy()
        color = (0, 200, 0) if matched else (0, 180, 255)
        label = f"similarity {similarity:.3f} {'MATCH' if matched else 'NO MATCH'}"
        cv2.putText(annotated, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        return (
            {
                "algorithm_id": spec.algorithm_id,
                "car_id": car_id,
                "stream_id": stream_id,
                "task": "similarity",
                "matched": matched,
                "similarity": round(similarity, 6),
                "threshold": threshold,
                "template_path": str(template_path),
            },
            annotated,
        )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / (norm_a * norm_b))))
