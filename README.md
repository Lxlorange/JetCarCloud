# JetCarCloud

JetCarCloud is the WSL/cloud-side service for JetCar. It handles app/edge
networking, image/video preprocessing, and algorithm dispatch. The current
runtime loads the three road-inspection models directly inside the WSL Python
process; Docker is no longer required for these algorithms.

API document: [docs/cloud_api.md](docs/cloud_api.md)

## Repository Layout

```text
JetCarCloud/
  algorithms.json           Runtime algorithm catalog
  app/
    main.py                 FastAPI routes
    config.py               Environment configuration
    schemas.py              Pydantic message models
    algorithms/
      catalog.py            Loads algorithm_id -> runner/model mapping
      local/                In-process YOLO/OpenCV algorithm implementations
      runner.py             Legacy Docker command runner
      service.py            Prepares input/output dirs and reads results
    video/
      processor.py          Frame decoding, preprocessing, JPEG encoding
      registry.py           Video stream registry and status
    inference/              Legacy in-process detector used by old websocket path
  scripts/
    register_video_stream.py
    run_algorithm.py
  models/
    yolov5-manhole/         Manhole YOLOv5 weights and source
    yolov8-road-damage/     Road-damage YOLOv8 weights
    yolov5-similarity/      Similarity template and weights
```

## Setup

```bash
cd /path/to/JetCarCloud
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Start:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

## Algorithm Catalog

The default catalog uses local runners:

```json
{
  "algorithms": {
    "yolov8-road-damage": {
      "name": "YOLOv8 Road Damage Detection",
      "runner": "local",
      "image": "",
      "inputs": ["frame.jpg", "request.json"],
      "outputs": ["result.json", "annotated.jpg"],
      "timeout_seconds": 60,
      "enabled": true,
      "metadata": {
        "task": "road_damage_detection",
        "model_path": "models/yolov8-road-damage/YOLOv8_Small_2nd_Model.pt",
        "device": "cpu"
      }
    }
  }
}
```

Reload after editing:

```bash
curl -X POST http://127.0.0.1:8000/api/algorithms/reload
```

List algorithms:

```bash
curl http://127.0.0.1:8000/api/algorithms
```

The current runtime catalog registers these local algorithms:

```text
yolov5-similarity       -> OpenCV feature similarity, template under models/yolov5-similarity/
yolov5-manhole-detect  -> YOLOv5, weights under models/yolov5-manhole/
yolov8-road-damage     -> Ultralytics YOLOv8, weights under models/yolov8-road-damage/
```

Adding another model should only require adding one entry to `algorithms.json`
and implementing a matching local runner under `app/algorithms/local/`.

## Run an Algorithm on an Uploaded Image

```bash
python scripts/run_algorithm.py \
  --cloud http://127.0.0.1:8000 \
  --algorithm-id yolov8-road-damage \
  --image /path/to/road-test.jpg \
  --include-image
```

API:

```text
POST /api/algorithms/{algorithm_id}/run
```

The response is:

```json
{
  "type": "algorithm_result",
  "ok": true,
  "algorithm_id": "yolov8-road-damage",
  "car_id": "car_001",
  "stream_id": "upload",
  "runner": "local",
  "latency_ms": 1234.5,
  "result": {},
  "outputs": {},
  "annotated_image": null,
  "error": ""
}
```

If `--include-image` is set and the local runner returns an annotated frame,
`annotated_image` contains a base64 JPEG `ImagePayload` that the mobile app can
display directly.

## Push Edge Camera Frames

In the real system, the edge side should push JPEG frames to Cloud over LAN.
Cloud auto-creates or updates the stream when the first frame arrives:

```text
POST /api/video/streams/{car_id}/{stream_id}/frames
```

To trigger one algorithm while pushing a frame:

```text
POST /api/video/streams/{car_id}/{stream_id}/frames?algorithm_id=yolov5-manhole-detect
```

To trigger multiple algorithms on the same frame:

```text
POST /api/video/streams/{car_id}/{stream_id}/frames?algorithm_ids=yolov5-manhole-detect,yolov8-road-damage
```

For WebSocket push, use the same query parameters:

```text
ws://<cloud-ip>:8000/ws/video/car_001/camera_front/edge?algorithm_ids=yolov5-manhole-detect,yolov8-road-damage
```

Realtime push is intentionally throttled. Cloud keeps the latest accepted frame,
but it does not run every incoming frame through every model:

```text
VIDEO_PUSH_MIN_INTERVAL_MS=200       # accept at most about 5 pushed frames/s per stream
ALGORITHM_MIN_INTERVAL_MS=1000       # start each algorithm at most about 1 time/s per stream
ALGORITHM_MAX_CONCURRENT_TASKS=2     # total algorithm tasks running at the same time
DEBUG_DUMP_SKIPPED_FRAMES=false      # skipped frames are not written to .jetcar_debug
```

If a frame is dropped by the input limiter, the upload response has
`frame_accepted=false`. If a frame is accepted but an algorithm is already busy
or rate-limited, `algorithms_skipped` explains the reason. Skipped frames are not
queued; this protects WSL from process buildup and keeps the service using the
newest frame.

Payload:

```json
{
  "car_id": "car_001",
  "image": {
    "encoding": "jpeg",
    "width": 1280,
    "height": 720,
    "data": "base64-jpeg"
  }
}
```

For local integration without the car, push a repeated mock frame:

```bash
python scripts/push_frame.py \
  --cloud http://127.0.0.1:8000 \
  --car-id car_001 \
  --stream-id camera_front \
  --image /path/to/mock-frame.jpg \
  --repeat \
  --fps 1
```

## Register Pull-Based Camera Streams

If the car exposes RTSP/MJPEG instead of pushing frames, register its LAN URL:

```bash
python scripts/register_video_stream.py \
  --cloud http://127.0.0.1:8000 \
  --car-id car_001 \
  --stream-id camera_front \
  --url rtsp://192.168.10.50:8554/camera \
  --transport rtsp \
  --width 640 \
  --height 640
```

Local file paths are only for development when no car is available.

## Run an Algorithm on One Video Frame

Run one frame from the latest pushed frame or from a registered pull stream:

```bash
python scripts/run_algorithm.py \
  --cloud http://127.0.0.1:8000 \
  --algorithm-id yolov8-road-damage \
  --car-id car_001 \
  --stream-id camera_front \
  --stream \
  --include-image
```

API:

```text
POST /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/run-once
```

## Stream Processed Frames to the App

JetCarCloud can expose a low-FPS MJPEG stream of algorithm-processed frames:

```text
GET /api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg?fps=1
```

Example:

```bash
curl -v \
  "http://127.0.0.1:8000/api/video/streams/car_001/camera_front/algorithms/yolov5-similarity/mjpeg?fps=1"
```

Open the same URL in a browser, VLC, or the mobile app image view. Each frame is
read from the latest pushed edge frame or captured from the registered camera
URL, processed by the configured local runner, and streamed back from the latest
cached annotated frame.

Each MJPEG URL shows one algorithm's processed frames. If the edge side triggers
both manhole and road-damage algorithms, open one URL per algorithm:

```text
http://<cloud-ip>:8000/api/video/streams/car_001/camera_front/algorithms/yolov5-manhole-detect/mjpeg?fps=5
http://<cloud-ip>:8000/api/video/streams/car_001/camera_front/algorithms/yolov8-road-damage/mjpeg?fps=5
```

This is intended for low-FPS preview. The local models are loaded once and then
reused, but the push limiter still controls how often inference is started.

## Update an Algorithm Model

Replace the local model file referenced by `algorithms.json`, then restart
JetCarCloud so the in-memory model is reloaded:

```text
models/yolov5-manhole/Manhole_model.pt
models/yolov8-road-damage/YOLOv8_Small_2nd_Model.pt
models/yolov5-similarity/target.jpg
```

If you only change `algorithms.json`, reload the catalog:

```bash
curl -X POST http://127.0.0.1:8000/api/algorithms/reload
```

## Video Stream Background Mode

Start/stop stream sampling:

```bash
curl -X POST http://127.0.0.1:8000/api/video/streams/car_001/camera_front/start
curl -X POST http://127.0.0.1:8000/api/video/streams/car_001/camera_front/stop
```

By default this only publishes `video_frame` metadata. To run algorithms in the
background, include algorithm IDs in the stream metadata when registering it:

```json
{
  "car_id": "car_001",
  "stream_id": "camera_front",
  "url": "rtsp://192.168.10.50:8554/camera",
  "transport": "rtsp",
  "metadata": {
    "algorithms": ["yolov8-road-damage"]
  }
}
```

## Windows WSL Port Forwarding

If the service runs inside WSL2 and the phone needs to access it through the
Windows host IP, run PowerShell as Administrator:

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=localhost connectport=8000
New-NetFirewallRule -DisplayName "JetCarCloud 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```
