from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import requests


def encode_image(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"failed to read image: {path}")
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise SystemExit(f"failed to encode image: {path}")
    return {
        "encoding": "jpeg",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


def post_json(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", default="http://127.0.0.1:8000")
    parser.add_argument("--car-id", default="car_001")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    base = args.cloud.rstrip("/")
    ref_payload = {"car_id": args.car_id, "image": encode_image(Path(args.reference))}
    query_payload = {"car_id": args.car_id, "image": encode_image(Path(args.query))}

    print(post_json(f"{base}/api/edge/reference", ref_payload))
    print(post_json(f"{base}/api/app/compare", query_payload))


if __name__ == "__main__":
    main()
