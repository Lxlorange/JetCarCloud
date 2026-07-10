from __future__ import annotations

import time

import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.config import get_settings
from app.connection_manager import ConnectionManager
from app.image_codec import decode_jpeg
from app.inference.detector import build_detector
from app.schemas import (
    EdgeFrame,
    ImageUpload,
    InferenceResult,
    ReferenceUploadResult,
    SimilarityResult,
)
from app.similarity import compare_images

settings = get_settings()
manager = ConnectionManager(history_size=settings.app_result_history)
detector = build_detector(
    settings.yolo_model_path,
    backend=settings.yolo_backend,
    yolov5_repo_path=settings.yolov5_repo_path,
    device=settings.yolo_device,
    confidence=settings.yolo_confidence,
    image_size=settings.yolo_image_size,
)

app = FastAPI(title=settings.app_name)
reference_images = {}


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.app_name,
        "detector": detector.__class__.__name__,
        "detector_reason": getattr(detector, "reason", ""),
        "connections": await manager.stats(),
        "edge_frame_url": settings.edge_frame_url,
    }


@app.post("/api/edge/reference", response_model=ReferenceUploadResult)
async def upload_reference(payload: ImageUpload) -> ReferenceUploadResult:
    image = decode_jpeg(payload.image)
    reference_images[payload.car_id] = image
    return ReferenceUploadResult(
        car_id=payload.car_id,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        message="reference image stored",
    )


@app.post("/api/app/compare", response_model=SimilarityResult)
async def compare_with_reference(payload: ImageUpload) -> SimilarityResult:
    reference = None
    reference_source = "cache"
    if settings.edge_frame_url:
        try:
            reference = fetch_edge_frame(settings.edge_frame_url)
            reference_source = "edge_frame_url"
        except Exception as exc:
            return SimilarityResult(
                ok=False,
                car_id=payload.car_id,
                similarity=0.0,
                matched=False,
                threshold=0.45,
                method="none",
                server_latency_ms=0.0,
                yolo_summary={
                    "error": f"failed to fetch edge frame from EDGE_FRAME_URL: {exc}",
                    "edge_frame_url": settings.edge_frame_url,
                },
                reference_source="edge_frame_url_error",
            )
    else:
        reference = reference_images.get(payload.car_id)

    if reference is None:
        error = "no reference image uploaded and EDGE_FRAME_URL is empty"
        return SimilarityResult(
            ok=False,
            car_id=payload.car_id,
            similarity=0.0,
            matched=False,
            threshold=0.45,
            method="none",
            server_latency_ms=0.0,
            yolo_summary={"error": error},
            reference_source="none",
        )

    query = decode_jpeg(payload.image)
    result = compare_images(reference, query, detector=detector, threshold=0.45)
    return SimilarityResult(
        car_id=payload.car_id,
        similarity=result.similarity,
        matched=result.matched,
        threshold=result.threshold,
        method=result.method,
        server_latency_ms=result.latency_ms,
        yolo_summary=result.yolo_summary,
        reference_source=reference_source,
    )


def fetch_edge_frame(url: str):
    response = requests.get(url, timeout=(3, 10))
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = ImageUpload.model_validate(response.json())
        return decode_jpeg(payload.image)

    import cv2
    import numpy as np

    array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"edge frame response is not a decodable image: {url}")
    return image


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
