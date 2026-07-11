from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "JetCarCloud"
    host: str = "0.0.0.0"
    port: int = 8000
    yolo_backend: str = "auto"
    yolo_model_path: str = ""
    yolov5_repo_path: str = ""
    yolo_device: str = "cpu"
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_image_size: int = Field(default=640, ge=64)
    app_result_history: int = Field(default=1, ge=0, le=10)
    edge_frame_url: str = ""
    video_default_width: int = Field(default=640, ge=64)
    video_default_height: int = Field(default=640, ge=64)
    video_default_fps: float = Field(default=2.0, ge=0.1, le=30.0)
    video_capture_timeout_ms: int = Field(default=5000, ge=100)
    manhole_provider: str = "local"
    manhole_model_path: str = ""
    manhole_backend: str = "yolov5"
    manhole_yolov5_repo_path: str = ""
    manhole_device: str = "cpu"
    manhole_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    manhole_image_size: int = Field(default=640, ge=64)
    manhole_positive_labels: str = "manhole,manhole-cover,cover"
    roboflow_api_key: str = ""
    roboflow_model_id: str = "manhole-wsmwd"
    roboflow_model_version: str = "1"
    roboflow_api_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
