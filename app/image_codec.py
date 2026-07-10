from __future__ import annotations

import base64

import cv2
import numpy as np

from app.schemas import ImagePayload


def decode_jpeg(payload: ImagePayload) -> np.ndarray:
    raw = base64.b64decode(payload.data)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid jpeg image payload")
    return image

