from __future__ import annotations

import time

import numpy as np

from app.features.annotate import draw_detection_groups
from app.features.manhole import ManholeProvider
from app.features.road_defect import RoadDefectProvider
from app.schemas import RoadInspectionResult
from app.video import encode_jpeg_payload


def inspect_road(
    image: np.ndarray,
    *,
    car_id: str,
    stream_id: str,
    manhole_provider: ManholeProvider,
    road_defect_provider: RoadDefectProvider,
    metadata: dict | None = None,
    include_image: bool = False,
) -> RoadInspectionResult:
    started = time.perf_counter()
    manhole = manhole_provider.detect(
        image,
        car_id=car_id,
        stream_id=stream_id,
        metadata=metadata,
        include_image=False,
    )
    road_defect = road_defect_provider.detect(
        image,
        car_id=car_id,
        stream_id=stream_id,
        metadata=metadata,
        include_image=False,
    )
    annotated = None
    if include_image:
        annotated_image = draw_detection_groups(
            image,
            [
                ("manhole:", manhole.detections, (0, 180, 255)),
                ("defect:", road_defect.detections, (0, 0, 255)),
            ],
        )
        annotated = encode_jpeg_payload(annotated_image)

    latency_ms = (time.perf_counter() - started) * 1000.0
    return RoadInspectionResult(
        ok=manhole.ok and road_defect.ok,
        car_id=car_id,
        stream_id=stream_id,
        latency_ms=round(latency_ms, 3),
        manhole=manhole,
        road_defect=road_defect,
        metadata=metadata or {},
        annotated_image=annotated,
    )
