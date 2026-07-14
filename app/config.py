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
    video_push_min_interval_ms: int = Field(default=250, ge=0)
    algorithm_min_interval_ms: int = Field(default=250, ge=0)
    algorithm_max_concurrent_tasks: int = Field(default=1, ge=1)
    algorithm_realtime_persist_outputs: bool = False
    algorithm_catalog_path: str = "algorithms.json"
    algorithm_work_dir: str = ".jetcar_algorithm_runs"
    docker_executable: str = "docker"
    debug_dump_enabled: bool = False
    debug_dump_dir: str = ".jetcar_debug"
    debug_save_images: bool = False
    debug_save_algorithm_outputs: bool = False
    debug_dump_skipped_frames: bool = False
    reports_dir: str = ".jetcar_reports"
    map_dir: str = ".jetcar_maps"
    discovery_beacon_enabled: bool = False
    discovery_beacon_port: int = Field(default=8765, ge=1, le=65535)
    discovery_beacon_interval_seconds: float = Field(default=1.0, ge=0.2)
    discovery_beacon_host: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
