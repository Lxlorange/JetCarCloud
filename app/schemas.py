from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImagePayload(BaseModel):
    encoding: Literal["jpeg"] = "jpeg"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    data: str = Field(min_length=1)


class EdgeFrame(BaseModel):
    type: Literal["edge_frame"]
    car_id: str = Field(min_length=1)
    timestamp: float
    image: ImagePayload
    sensors: dict = Field(default_factory=dict)


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    distance_m: float | None = None


class InferenceResult(BaseModel):
    type: Literal["yolo_fusion"] = "yolo_fusion"
    car_id: str
    edge_timestamp: float
    server_latency_ms: float
    detections: list[Detection] = Field(default_factory=list)

