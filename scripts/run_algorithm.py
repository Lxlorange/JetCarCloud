from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured JetCarCloud algorithm by algorithm_id.")
    parser.add_argument("--cloud", default="http://127.0.0.1:8000")
    parser.add_argument("--algorithm-id", required=True)
    parser.add_argument("--car-id", default="car_001")
    parser.add_argument("--stream-id", default="camera_front")
    parser.add_argument("--image", help="Local image path for upload-based algorithm execution.")
    parser.add_argument("--stream", action="store_true", help="Run algorithm once on a registered video stream.")
    parser.add_argument("--include-image", action="store_true", help="Ask cloud to return output/annotated.jpg as ImagePayload.")
    args = parser.parse_args()

    base_url = args.cloud.rstrip("/")
    if args.stream:
        url = f"{base_url}/api/video/streams/{args.car_id}/{args.stream_id}/algorithms/{args.algorithm_id}/run-once"
        response = requests.post(url, params={"include_image": args.include_image}, timeout=120)
    else:
        if not args.image:
            raise SystemExit("--image is required unless --stream is set")
        payload = _image_payload(Path(args.image), args.car_id, args.stream_id, include_image=args.include_image)
        response = requests.post(f"{base_url}/api/algorithms/{args.algorithm_id}/run", json=payload, timeout=120)

    response.raise_for_status()
    print(response.json())


def _image_payload(path: Path, car_id: str, stream_id: str, *, include_image: bool) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"failed to read image: {path}")
    height, width = image.shape[:2]
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise SystemExit(f"failed to encode image: {path}")
    return {
        "car_id": car_id,
        "stream_id": stream_id,
        "include_image": include_image,
        "image": {
            "encoding": "jpeg",
            "width": int(width),
            "height": int(height),
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
        "parameters": {},
    }


if __name__ == "__main__":
    main()
