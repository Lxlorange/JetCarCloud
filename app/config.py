from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "JetCarCloud"
    host: str = "0.0.0.0"
    port: int = 8000
    yolo_model_path: str = ""
    yolo_device: str = "cpu"
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_image_size: int = Field(default=640, ge=64)
    app_result_history: int = Field(default=1, ge=0, le=10)
    edge_frame_url: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
