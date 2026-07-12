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
    algorithm_catalog_path: str = "algorithms.json"
    algorithm_work_dir: str = ".jetcar_algorithm_runs"
    docker_executable: str = "docker"
    debug_dump_enabled: bool = True
    debug_dump_dir: str = ".jetcar_debug"
    debug_save_images: bool = True
    debug_save_algorithm_outputs: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
