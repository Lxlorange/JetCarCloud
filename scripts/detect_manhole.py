from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Call JetCarCloud manhole detection APIs.")
    parser.add_argument("--cloud", default="http://127.0.0.1:8000")
    parser.add_argument("--car-id", default="car_001")
    parser.add_argument("--stream-id", default="camera_front")
    parser.add_argument("--image", help="Local image path for /api/features/manhole/detect.")
    parser.add_argument("--stream", action="store_true", help="Run detection once on a registered video stream.")
    args = parser.parse_args()

    base_url = args.cloud.rstrip("/")
    if args.stream:
        url = f"{base_url}/api/video/streams/{args.car_id}/{args.stream_id}/features/manhole/run-once"
        response = requests.post(url, timeout=60)
    else:
        if not args.image:
            raise SystemExit("--image is required unless --stream is set")
        payload = _image_payload(Path(args.image), args.car_id)
        response = requests.post(f"{base_url}/api/features/manhole/detect", json=payload, timeout=60)

    response.raise_for_status()
    print(response.json())


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
