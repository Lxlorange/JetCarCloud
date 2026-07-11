from __future__ import annotations

import time
from dataclasses import dataclass

from app.schemas import VideoStreamConfig, VideoStreamStatus


@dataclass
class VideoStreamRuntime:
    config: VideoStreamConfig
    running: bool = False
    frame_count: int = 0
    last_error: str = ""
    last_frame_at: float | None = None


class VideoStreamRegistry:
    def __init__(self) -> None:
        self._streams: dict[tuple[str, str], VideoStreamRuntime] = {}

    def upsert(self, config: VideoStreamConfig) -> VideoStreamStatus:
        key = (config.car_id, config.stream_id)
        current = self._streams.get(key)
        if current is None:
            current = VideoStreamRuntime(config=config)
            self._streams[key] = current
        else:
            current.config = config
            current.last_error = ""
        return self.status(config.car_id, config.stream_id)

    def get(self, car_id: str, stream_id: str) -> VideoStreamRuntime | None:
        return self._streams.get((car_id, stream_id))

    def list(self) -> list[VideoStreamStatus]:
        return [self._to_status(runtime) for runtime in self._streams.values()]

    def status(self, car_id: str, stream_id: str) -> VideoStreamStatus:
        runtime = self._streams[(car_id, stream_id)]
        return self._to_status(runtime)

    def mark_started(self, car_id: str, stream_id: str) -> None:
        runtime = self._streams[(car_id, stream_id)]
        runtime.running = True
        runtime.last_error = ""

    def mark_stopped(self, car_id: str, stream_id: str, *, error: str = "") -> None:
        runtime = self._streams[(car_id, stream_id)]
        runtime.running = False
        runtime.last_error = error

    def mark_frame(self, car_id: str, stream_id: str) -> None:
        runtime = self._streams[(car_id, stream_id)]
        runtime.frame_count += 1
        runtime.last_frame_at = time.time()
        runtime.last_error = ""

    def mark_error(self, car_id: str, stream_id: str, error: str) -> None:
        runtime = self._streams[(car_id, stream_id)]
        runtime.last_error = error

    def _to_status(self, runtime: VideoStreamRuntime) -> VideoStreamStatus:
        config = runtime.config
        return VideoStreamStatus(
            car_id=config.car_id,
            stream_id=config.stream_id,
            url=config.url,
            transport=config.transport,
            enabled=config.enabled,
            running=runtime.running,
            frame_count=runtime.frame_count,
            last_error=runtime.last_error,
            last_frame_at=runtime.last_frame_at,
        )
