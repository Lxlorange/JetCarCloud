from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.algorithms.local.runner import resolve_project_path
from app.schemas import AlgorithmInfo
from app.similarity import extract_feature_vector


def extract_similarity_feature(image: np.ndarray) -> np.ndarray:
    return extract_feature_vector(image)


def save_similarity_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), feature)


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
        feature_path = parameters.get("feature_path")
        if feature_path:
            template_feature = np.load(str(feature_path))
        else:
            template_feature = extract_feature_vector(template)
        similarity = _cosine_similarity(source_feature, template_feature)
        matched = similarity >= threshold
        localization = _localize_template(image, template)

        annotated = image.copy()
        color = (0, 200, 0) if matched else (0, 180, 255)
        label = f"similarity {similarity:.3f} {'MATCH' if matched else 'NO MATCH'}"
        cv2.putText(annotated, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        if matched and localization is not None:
            x1, y1, x2, y2 = localization["bbox_px"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cx, cy = localization["center_px"]
            cv2.circle(annotated, (cx, cy), 5, color, -1)

        center_norm = localization["center_norm"] if localization is not None else [0.5, 0.5]
        bbox_norm = localization["bbox_norm"] if localization is not None else [0.0, 0.0, 1.0, 1.0]

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
                "center_norm": center_norm,
                "bbox_norm": bbox_norm,
                "localization": localization or {"method": "none", "matched_keypoints": 0},
                "control_hint": "target_found" if matched else "keep_searching",
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


def _localize_template(image: np.ndarray, template: np.ndarray) -> dict | None:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=800)
    template_keypoints, template_desc = orb.detectAndCompute(template_gray, None)
    image_keypoints, image_desc = orb.detectAndCompute(image_gray, None)
    if template_desc is None or image_desc is None:
        return None
    if len(template_keypoints) < 8 or len(image_keypoints) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(template_desc, image_desc, k=2)
    good = []
    for pair in matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good.append(first)
    if len(good) < 8:
        return None

    source_points = np.float32([template_keypoints[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    target_points = np.float32([image_keypoints[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return None

    height, width = template_gray.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    image_height, image_width = image.shape[:2]
    x1 = int(max(0, min(image_width - 1, np.min(projected[:, 0]))))
    y1 = int(max(0, min(image_height - 1, np.min(projected[:, 1]))))
    x2 = int(max(0, min(image_width - 1, np.max(projected[:, 0]))))
    y2 = int(max(0, min(image_height - 1, np.max(projected[:, 1]))))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return {
        "method": "orb_homography",
        "matched_keypoints": int(mask.sum()),
        "center_px": [center_x, center_y],
        "center_norm": [round(center_x / image_width, 6), round(center_y / image_height, 6)],
        "bbox_px": [x1, y1, x2, y2],
        "bbox_norm": [
            round(x1 / image_width, 6),
            round(y1 / image_height, 6),
            round(x2 / image_width, 6),
            round(y2 / image_height, 6),
        ],
    }
