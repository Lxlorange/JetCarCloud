from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import cv2
import websockets


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--car-id", default="car_001")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"failed to read image: {image_path}")

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise SystemExit("failed to encode image")

    payload = {
        "type": "edge_frame",
        "car_id": args.car_id,
        "timestamp": time.time(),
        "image": {
            "encoding": "jpeg",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        },
        "sensors": {
            "lidar": None,
            "imu": None,
        },
    }

    async with websockets.connect(args.url, max_size=16 * 1024 * 1024) as ws:
        await ws.send(json.dumps(payload, separators=(",", ":")))
        response = await ws.recv()
        print(response)


if __name__ == "__main__":
    asyncio.run(main())

