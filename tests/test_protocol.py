from __future__ import annotations

import numpy as np

from app.schemas import (
    AiTaskResult,
    Detection,
    EdgeFrame,
    InferenceResult,
    ManholeDetectionResult,
    RoadDefectDetectionResult,
    RoadInspectionResult,
    VideoStreamConfig,
)
from app.video import encode_jpeg_payload, preprocess_frame


def test_edge_frame_schema_accepts_minimal_payload() -> None:
    frame = EdgeFrame.model_validate(
        {
            "type": "edge_frame",
            "car_id": "car_001",
            "timestamp": 1.0,
            "image": {
                "encoding": "jpeg",
                "width": 640,
                "height": 480,
                "data": "abc",
            },
            "sensors": {},
        }
    )
    assert frame.car_id == "car_001"


def test_inference_result_schema() -> None:
    result = InferenceResult(
        car_id="car_001",
        edge_timestamp=1.0,
        server_latency_ms=5.0,
        detections=[
            Detection(
                label="person",
                confidence=0.9,
                bbox=[1, 2, 3, 4],
                distance_m=2.0,
            )
        ],
    )
    assert result.model_dump()["type"] == "yolo_fusion"


def test_video_stream_schema_defaults() -> None:
    config = VideoStreamConfig(url="rtsp://127.0.0.1:8554/camera")
    assert config.car_id == "car_001"
    assert config.stream_id == "camera_front"
    assert config.width == 640
    assert config.height == 640


def test_video_frame_preprocess_letterboxes_to_target_size() -> None:
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    processed = preprocess_frame(frame, width=640, height=640, source="test")
    payload = encode_jpeg_payload(processed.image)
    assert processed.image.shape == (640, 640, 3)
    assert processed.metadata.width == 320
    assert processed.metadata.height == 120
    assert payload.width == 640
    assert payload.height == 640


def test_ai_task_result_has_message_type() -> None:
    result = AiTaskResult(task_id="yolo_detection", car_id="car_001", latency_ms=1.0)
    assert result.model_dump()["type"] == "ai_task_result"


def test_manhole_detection_result_schema() -> None:
    result = ManholeDetectionResult(
        car_id="car_001",
        stream_id="camera_front",
        provider="local",
        found=True,
        count=1,
        detections=[Detection(label="manhole", confidence=0.8, bbox=[1, 2, 3, 4])],
    )
    data = result.model_dump()
    assert data["type"] == "manhole_detection"
    assert data["found"] is True
    assert data["count"] == 1


def test_road_defect_detection_result_schema() -> None:
    result = RoadDefectDetectionResult(
        car_id="car_001",
        stream_id="camera_front",
        provider="local",
        found=True,
        count=1,
        detections=[Detection(label="crack", confidence=0.8, bbox=[1, 2, 3, 4])],
    )
    data = result.model_dump()
    assert data["type"] == "road_defect_detection"
    assert data["found"] is True


def test_road_inspection_result_schema() -> None:
    manhole = ManholeDetectionResult(car_id="car_001", provider="local")
    road_defect = RoadDefectDetectionResult(car_id="car_001", provider="local")
    result = RoadInspectionResult(
        car_id="car_001",
        stream_id="camera_front",
        manhole=manhole,
        road_defect=road_defect,
    )
    assert result.model_dump()["type"] == "road_inspection"
