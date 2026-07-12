# JetCarCloud

JetCarCloud is the WSL/cloud-side service for JetCar. It handles app/edge
networking, image/video preprocessing, and algorithm dispatch. Model code is not
loaded directly by the cloud service anymore. Each model should be packaged as a
Docker image and registered in `algorithms.json`.

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

## Run an Algorithm on a Video Stream Frame

Register a stream:

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

Run one frame:

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
