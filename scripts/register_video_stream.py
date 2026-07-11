from __future__ import annotations

import argparse

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a mock JetCar video stream in JetCarCloud.")
    parser.add_argument("--cloud", default="http://127.0.0.1:8000")
    parser.add_argument("--car-id", default="car_001")
    parser.add_argument("--stream-id", default="camera_front")
    parser.add_argument("--url", required=True, help="RTSP/HTTP/file URL readable by the cloud process.")
    parser.add_argument("--transport", default="unknown", choices=["rtsp", "http_mjpeg", "http_file", "file", "unknown"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--sample-interval-ms", type=int, default=500)
    args = parser.parse_args()

    payload = {
        "car_id": args.car_id,
        "stream_id": args.stream_id,
        "url": args.url,
        "transport": args.transport,
        "width": args.width,
        "height": args.height,
        "sample_interval_ms": args.sample_interval_ms,
    }
    response = requests.post(f"{args.cloud.rstrip('/')}/api/video/streams", json=payload, timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
