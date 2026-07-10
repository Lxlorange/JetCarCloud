from __future__ import annotations

from app.schemas import Detection, EdgeFrame, InferenceResult


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

