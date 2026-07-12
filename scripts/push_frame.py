from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import cv2
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Push JPEG frames to JetCarCloud like an edge camera.")
    parser.add_argument("--cloud", default="http://127.0.0.1:8000")
    parser.add_argument("--car-id", default="car_001")
    parser.add_argument("--stream-id", default="camera_front")
    parser.add_argument("--image", required=True, help="Local image used as a mock camera frame.")
    parser.add_argument("--repeat", action="store_true", help="Push the same image continuously.")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    payload = _image_payload(Path(args.image), args.car_id)
    url = f"{args.cloud.rstrip('/')}/api/video/streams/{args.car_id}/{args.stream_id}/frames"

    while True:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(response.json())
        if not args.repeat:
            return
        time.sleep(max(0.05, 1.0 / args.fps))


def _image_payload(path: Path, car_id: str) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"failed to read image: {path}")
    height, width = image.shape[:2]
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit(f"failed to encode image: {path}")
    return {
        "car_id": car_id,
        "image": {
            "encoding": "jpeg",
            "width": int(width),
            "height": int(height),
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
    }


if __name__ == "__main__":
    main()
