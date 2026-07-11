from __future__ import annotations

import base64
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas import ImagePayload, VideoFrameMetadata


@dataclass(frozen=True)
class VideoFrame:
    image: np.ndarray
    metadata: VideoFrameMetadata


def open_stream_frame(
    url: str,
    *,
    width: int,
    height: int,
    timeout_ms: int = 5000,
    source: str = "stream",
) -> VideoFrame:
    capture = cv2.VideoCapture(url)
    try:
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    except Exception:
        pass

    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"failed to read a frame from video source: {url}")
        return preprocess_frame(frame, width=width, height=height, source=source)
    finally:
        capture.release()


def decode_video_chunk(
    data: str,
    *,
    encoding: str,
    frame_index: int,
    width: int,
    height: int,
    source: str = "upload",
) -> VideoFrame:
    suffix = _suffix_for_encoding(encoding)
    raw = base64.b64decode(data)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        temp_path = Path(tmp.name)

    capture = cv2.VideoCapture(str(temp_path))
    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"failed to decode frame {frame_index} from uploaded video chunk")
        return preprocess_frame(frame, width=width, height=height, source=source)
    finally:
        capture.release()
        try:
            temp_path.unlink()
        except OSError:
            pass


def preprocess_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    source: str,
    letterbox: bool = True,
) -> VideoFrame:
    if frame.ndim != 3:
        raise ValueError("video frame must be a color image")

    original_height, original_width, channels = frame.shape
    if letterbox:
        processed = _letterbox(frame, width=width, height=height)
    else:
        processed = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    metadata = VideoFrameMetadata(
        width=int(original_width),
        height=int(original_height),
        channels=int(channels),
        resized_width=int(processed.shape[1]),
        resized_height=int(processed.shape[0]),
        letterboxed=letterbox,
        source=source,
        timestamp=time.time(),
    )
    return VideoFrame(image=processed, metadata=metadata)


def encode_jpeg_payload(frame: np.ndarray, *, quality: int = 85) -> ImagePayload:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode frame as jpeg")
    height, width = frame.shape[:2]
    return ImagePayload(
        encoding="jpeg",
        width=int(width),
        height=int(height),
        data=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


def _letterbox(frame: np.ndarray, *, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def _suffix_for_encoding(encoding: str) -> str:
    if encoding in {"mp4", "avi", "mov", "mjpeg"}:
        return f".{encoding}"
    return ".bin"
