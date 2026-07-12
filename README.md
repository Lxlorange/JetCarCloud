# JetCarCloud

JetCarCloud is the WSL/cloud-side service for JetCar. It handles app/edge
networking, image/video preprocessing, and algorithm dispatch. Model code is not
loaded directly by the cloud service anymore. Each model should be packaged as a
Docker image and registered in `algorithms.json`.

API document: [docs/cloud_api.md](docs/cloud_api.md)

## Repository Layout

```text
JetCarCloud/
  algorithms.json           Empty runtime algorithm catalog
  algorithms.example.json   Example catalog entries
  app/
    main.py                 FastAPI routes
    config.py               Environment configuration
    schemas.py              Pydantic message models
    algorithms/
      catalog.py            Loads algorithm_id -> image/io mapping
      runner.py             Docker command runner
      service.py            Prepares input/output dirs and reads results
    video/
      processor.py          Frame decoding, preprocessing, JPEG encoding
      registry.py           Video stream registry and status
    inference/              Legacy in-process detector used by old websocket path
  scripts/
    register_video_stream.py
    run_algorithm.py
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

`algorithms.json` is intentionally empty by default:

```json
{
  "algorithms": {}
}
```

Add model containers by copying entries from `algorithms.example.json`:

```json
{
  "algorithms": {
    "road-inspection": {
      "name": "Road Inspection",
      "runner": "docker",
      "image": "jetcar-road-inspection:v1",
      "inputs": ["frame.jpg", "request.json"],
      "outputs": ["result.json", "annotated.jpg"],
      "timeout_seconds": 90,
      "enabled": true,
      "metadata": {
        "task": "run multiple road inspection models in one container"
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

## Build the YOLOv5 Similarity Image in WSL

From WSL:

```bash
cd /mnt/d/2026_spring/car/model-yolov5-similarity
docker build -t model-yolov5-similarity:v1 .
```

Register it in `JetCarCloud/algorithms.json`:

```json
{
  "algorithms": {
    "yolov5-similarity": {
      "name": "YOLOv5 Similarity",
      "runner": "docker",
      "image": "model-yolov5-similarity:v1",
      "inputs": ["frame.jpg", "request.json"],
      "outputs": ["result.json", "annotated.jpg"],
      "timeout_seconds": 60,
      "enabled": true,
      "metadata": {}
    }
  }
}
```

Reload the catalog:

```bash
curl -X POST http://127.0.0.1:8000/api/algorithms/reload
```

The current runtime catalog registers these Docker images:

```text
yolov5-similarity       -> model-yolov5-similarity:v1
yolov5-manhole-detect  -> model-yolov5-manhole-detect:v1
yolov8-road-damage     -> model-yolov8-road-damage:v1
```

Adding another model should only require adding one entry to `algorithms.json`
as long as the container follows the contract below.

## Container Contract

Cloud runs containers like this:

```bash
docker run --rm \
  -v "/host/run/input:/app/data/input" \
  -v "/host/run/output:/app/data/output" \
  your-image:v1
```

Each algorithm container should read:

```text
/app/data/input/frame.jpg
/app/data/input/request.json
```

And should write:

```text
/app/data/output/result.json
/app/data/output/annotated.jpg        optional
```

`request.json` contains:

```json
{
  "algorithm_id": "road-inspection",
  "car_id": "car_001",
  "stream_id": "camera_front",
  "parameters": {},
  "input_dir": "/app/data/input",
  "output_dir": "/app/data/output"
}
```

`result.json` can use any algorithm-specific JSON shape. JetCarCloud returns it
under the `result` field without interpreting it.

## Run an Algorithm on an Uploaded Image

```bash
python scripts/run_algorithm.py \
  --cloud http://127.0.0.1:8000 \
  --algorithm-id road-inspection \
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
  "algorithm_id": "road-inspection",
  "car_id": "car_001",
  "stream_id": "upload",
  "runner": "docker",
  "latency_ms": 1234.5,
  "result": {},
  "outputs": {},
  "annotated_image": null,
  "error": ""
}
```

If `--include-image` is set and the container writes `annotated.jpg`,
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
  --algorithm-id road-inspection \
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
URL, written to `/app/data/input/frame.jpg`, processed by the configured Docker
image, and streamed back from `/app/data/output/annotated.jpg`.

Each MJPEG URL shows one algorithm's processed frames. If the edge side triggers
both manhole and road-damage algorithms, open one URL per algorithm:

```text
http://<cloud-ip>:8000/api/video/streams/car_001/camera_front/algorithms/yolov5-manhole-detect/mjpeg?fps=5
http://<cloud-ip>:8000/api/video/streams/car_001/camera_front/algorithms/yolov8-road-damage/mjpeg?fps=5
```

This is functional for preview and integration testing, but it starts a Docker
container per frame. For higher real-time FPS, package the model as a persistent
HTTP/gRPC service container and add a long-running runner later.

## Update an Algorithm Image

Rebuild with the same tag:

```bash
cd /mnt/d/2026_spring/car/model-yolov5-similarity
docker build -t model-yolov5-similarity:v1 .
```

If you change the image tag, update `JetCarCloud/algorithms.json` and reload:

```bash
curl -X POST http://127.0.0.1:8000/api/algorithms/reload
```

If Docker cache keeps an old layer during development:

```bash
docker build --no-cache -t model-yolov5-similarity:v1 .
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
    "algorithms": ["road-inspection"]
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
