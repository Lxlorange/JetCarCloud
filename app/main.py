from __future__ import annotations

import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import get_settings
from app.connection_manager import ConnectionManager
from app.image_codec import decode_jpeg
from app.inference.detector import build_detector
from app.schemas import EdgeFrame, InferenceResult

settings = get_settings()
manager = ConnectionManager(history_size=settings.app_result_history)
detector = build_detector(
    settings.yolo_model_path,
    device=settings.yolo_device,
    confidence=settings.yolo_confidence,
    image_size=settings.yolo_image_size,
)

app = FastAPI(title=settings.app_name)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.app_name,
        "detector": detector.__class__.__name__,
        "connections": await manager.stats(),
    }


@app.websocket("/ws/inference/{car_id}/edge")
async def edge_socket(websocket: WebSocket, car_id: str) -> None:
    await websocket.accept()
    while True:
        try:
            payload = await websocket.receive_json()
            frame = EdgeFrame.model_validate(payload)
            if frame.car_id != car_id:
                await websocket.send_json({"type": "error", "message": "car_id mismatch"})
                continue

            result = run_inference(frame)
            data = result.model_dump(mode="json")
            await websocket.send_json(data)
            await manager.publish(car_id, data)
        except WebSocketDisconnect:
            return
        except ValidationError as exc:
            await websocket.send_json({"type": "error", "message": exc.errors()})
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})


@app.websocket("/ws/inference/{car_id}/app")
async def app_socket(websocket: WebSocket, car_id: str) -> None:
    await manager.connect_app(car_id, websocket)
    try:
        while True:
            # Keep the socket open and accept optional pings or client commands.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect_app(car_id, websocket)


def run_inference(frame: EdgeFrame) -> InferenceResult:
    started = time.perf_counter()
    image = decode_jpeg(frame.image)
    detections = detector.detect(image, frame.sensors)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return InferenceResult(
        car_id=frame.car_id,
        edge_timestamp=frame.timestamp,
        server_latency_ms=round(latency_ms, 3),
        detections=detections,
    )

