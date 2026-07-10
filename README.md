# JetCarCloud

JetCarCloud is the cloud/WSL-side inference service for JetCar. It receives
camera frames from the Jetson edge node, runs object detection, and distributes
the result to both the edge node and any app clients subscribed to the same
`car_id`.

## Technology Choice

- Runtime: WSL2 Ubuntu 22.04 or a Linux server.
- Web framework: FastAPI, because it provides HTTP health checks and native
  WebSocket handling with a small amount of code.
- Inference: Ultralytics YOLO when a model path is configured. The service also
  has a no-op detector fallback so networking can be tested before model setup.
- Image processing: OpenCV + NumPy for base64 JPEG decoding.
- Configuration: environment variables and `.env.example`.

The service does not store data by default. It acts as a live inference relay.

## Repository Layout

```text
JetCarCloud/
  app/
    main.py                 FastAPI app and WebSocket routes
    config.py               Environment configuration
    connection_manager.py   App subscriber management
    schemas.py              Pydantic message models
    image_codec.py          Base64 JPEG decoder
    inference/
      detector.py           Detector interface and YOLO implementation
      fusion.py             Distance estimation hook
  scripts/
    send_test_frame.py      Sends one local image through the edge WebSocket
  tests/
    test_protocol.py        Lightweight schema checks
  .env.example
  requirements.txt
```

## Environment Commands To Run

Run these in WSL2 Ubuntu or a Linux server:

```bash
cd /path/to/JetCarCloud
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

If you already have a YOLOv5 or YOLOv8 weight file, edit `.env`:

```bash
YOLO_MODEL_PATH=/path/to/best.pt
YOLO_DEVICE=cpu
```

Start the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

If this service is running inside WSL2 and Jetson/App need to access it through
the Windows host IP, run PowerShell as Administrator on Windows:

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=localhost connectport=8000
New-NetFirewallRule -DisplayName "JetCarCloud 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## WebSocket Endpoints

Jetson edge upload:

```text
ws://{cloud-host}:8000/ws/inference/{car_id}/edge
```

Flutter app result subscription:

```text
ws://{cloud-host}:8000/ws/inference/{car_id}/app
```

The edge connection sends an `edge_frame` JSON message. The service replies to
that edge connection and broadcasts the same `yolo_fusion` result to all app
subscribers for that `car_id`.

## Test One Image

After the server starts, send a local image:

```bash
python scripts/send_test_frame.py \
  --url ws://127.0.0.1:8000/ws/inference/car_001/edge \
  --image /path/to/test.jpg
```

## Similarity Workflow

1. Start JetCarCloud.
2. Upload a simulated camera/reference frame from JetCarEdge.
3. In the mobile app, open the third tab, choose a gallery image, and upload it
   for comparison.

HTTP APIs:

```text
POST /api/edge/reference
POST /api/app/compare
```

Both accept this JSON shape:

```json
{
  "car_id": "car_001",
  "image": {
    "encoding": "jpeg",
    "width": 640,
    "height": 480,
    "data": "base64-jpeg"
  }
}
```

The comparison response includes `similarity`, `matched`, `threshold`,
`server_latency_ms`, and a YOLO label summary. If no `YOLO_MODEL_PATH` is set,
the service still runs OpenCV feature cosine similarity and reports YOLO as
unavailable.
