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


class ImageUpload(BaseModel):
    car_id: str = Field(default="car_001", min_length=1)
    image: ImagePayload


class ReferenceUploadResult(BaseModel):
    ok: bool = True
    car_id: str
    width: int
    height: int
    message: str


class SimilarityResult(BaseModel):
    ok: bool = True
    car_id: str
    similarity: float
    matched: bool
    threshold: float
    method: str
    server_latency_ms: float
    yolo_summary: dict = Field(default_factory=dict)
    reference_source: str = "cache"


class VideoStreamConfig(BaseModel):
    car_id: str = Field(default="car_001", min_length=1)
    stream_id: str = Field(default="camera_front", min_length=1)
    url: str = Field(min_length=1)
    transport: Literal["rtsp", "http_mjpeg", "http_file", "file", "unknown"] = "unknown"
    width: int = Field(default=640, ge=64)
    height: int = Field(default=640, ge=64)
    fps: float = Field(default=2.0, ge=0.1, le=30.0)
    sample_interval_ms: int = Field(default=500, ge=33)
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


class VideoStreamStatus(BaseModel):
    ok: bool = True
    car_id: str
    stream_id: str
    url: str
    transport: str
    enabled: bool
    running: bool = False
    frame_count: int = 0
    last_error: str = ""
    last_frame_at: float | None = None


class VideoFrameMetadata(BaseModel):
    width: int
    height: int
    channels: int
    resized_width: int
    resized_height: int
    letterboxed: bool
    source: str
    timestamp: float


class VideoFramePreprocessResult(BaseModel):
    ok: bool = True
    car_id: str
    stream_id: str = "camera_front"
    frame: ImagePayload
    metadata: VideoFrameMetadata


class VideoChunkUpload(BaseModel):
    car_id: str = Field(default="car_001", min_length=1)
    stream_id: str = Field(default="camera_front", min_length=1)
    encoding: Literal["mp4", "avi", "mov", "mjpeg", "unknown"] = "unknown"
    data: str = Field(min_length=1)
    frame_index: int = Field(default=0, ge=0)


class AiTaskSpec(BaseModel):
    task_id: str = Field(min_length=1)
    kind: Literal["yolo", "docker", "custom"] = "yolo"
    model_path: str = ""
    backend: str = "auto"
    docker_image: str = ""
    docker_command: list[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


class AiTaskResult(BaseModel):
    type: Literal["ai_task_result"] = "ai_task_result"
    ok: bool = True
    task_id: str
    car_id: str
    stream_id: str = "camera_front"
    latency_ms: float
    detections: list[Detection] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ManholeDetectionResult(BaseModel):
    type: Literal["manhole_detection"] = "manhole_detection"
    ok: bool = True
    car_id: str = "car_001"
    stream_id: str = "camera_front"
    provider: Literal["local", "roboflow", "none"] = "none"
    found: bool = False
    count: int = 0
    latency_ms: float = 0.0
    detections: list[Detection] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    error: str = ""
