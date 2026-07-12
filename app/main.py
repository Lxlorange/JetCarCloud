from __future__ import annotations

import asyncio
import time

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.algorithms import AlgorithmCatalog, AlgorithmService
from app.config import get_settings
from app.connection_manager import ConnectionManager
from app.image_codec import decode_jpeg
from app.inference.detector import build_detector
from app.schemas import (
    AlgorithmInfo,
    AlgorithmRunRequest,
    AlgorithmRunResult,
    EdgeFrame,
    ImageUpload,
    InferenceResult,
    ReferenceUploadResult,
    SimilarityResult,
    VideoChunkUpload,
    VideoFramePreprocessResult,
    VideoStreamConfig,
    VideoStreamStatus,
)
from app.similarity import compare_images
from app.video import (
    VideoStreamRegistry,
    decode_video_chunk,
    encode_jpeg_payload,
    open_stream_frame,
)

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
algorithm_service = AlgorithmService(
    catalog=AlgorithmCatalog(settings.algorithm_catalog_path),
    work_dir=settings.algorithm_work_dir,
    docker_executable=settings.docker_executable,
)

app = FastAPI(title=settings.app_name)
reference_images = {}
video_streams = VideoStreamRegistry()
stream_workers: dict[tuple[str, str], asyncio.Task] = {}


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": settings.app_name,
        "detector": detector.__class__.__name__,
        "detector_reason": getattr(detector, "reason", ""),
        "connections": await manager.stats(),
        "edge_frame_url": settings.edge_frame_url,
        "video_streams": len(video_streams.list()),
        "algorithms": [item.algorithm_id for item in algorithm_service.catalog.list()],
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


def _read_registered_stream_frame(car_id: str, stream_id: str):
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")

    config = runtime.config
    try:
        frame = open_stream_frame(
            config.url,
            width=config.width,
            height=config.height,
            timeout_ms=settings.video_capture_timeout_ms,
            source=f"{config.transport}:{config.stream_id}",
        )
        video_streams.mark_frame(car_id, stream_id)
        return frame
    except Exception as exc:
        video_streams.mark_error(car_id, stream_id, str(exc))
        raise HTTPException(status_code=502, detail=f"failed to read stream frame: {exc}") from exc


@app.post("/api/video/streams", response_model=VideoStreamStatus)
async def register_video_stream(payload: VideoStreamConfig) -> VideoStreamStatus:
    config = payload.model_copy(
        update={
            "width": payload.width or settings.video_default_width,
            "height": payload.height or settings.video_default_height,
        }
    )
    return video_streams.upsert(config)


@app.get("/api/video/streams", response_model=list[VideoStreamStatus])
async def list_video_streams() -> list[VideoStreamStatus]:
    return video_streams.list()


@app.get("/api/video/streams/{car_id}/{stream_id}", response_model=VideoStreamStatus)
async def get_video_stream(car_id: str, stream_id: str) -> VideoStreamStatus:
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")
    return video_streams.status(car_id, stream_id)


@app.post("/api/video/streams/{car_id}/{stream_id}/preprocess", response_model=VideoFramePreprocessResult)
async def preprocess_stream_frame(car_id: str, stream_id: str) -> VideoFramePreprocessResult:
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")

    config = runtime.config
    try:
        frame = open_stream_frame(
            config.url,
            width=config.width,
            height=config.height,
            timeout_ms=settings.video_capture_timeout_ms,
            source=f"{config.transport}:{config.stream_id}",
        )
        video_streams.mark_frame(car_id, stream_id)
    except Exception as exc:
        video_streams.mark_error(car_id, stream_id, str(exc))
        raise HTTPException(status_code=502, detail=f"failed to preprocess stream frame: {exc}") from exc

    return VideoFramePreprocessResult(
        car_id=car_id,
        stream_id=stream_id,
        frame=encode_jpeg_payload(frame.image),
        metadata=frame.metadata,
    )


@app.post("/api/video/chunks/preprocess", response_model=VideoFramePreprocessResult)
async def preprocess_video_chunk(payload: VideoChunkUpload) -> VideoFramePreprocessResult:
    runtime = video_streams.get(payload.car_id, payload.stream_id)
    width = runtime.config.width if runtime else settings.video_default_width
    height = runtime.config.height if runtime else settings.video_default_height

    try:
        frame = decode_video_chunk(
            payload.data,
            encoding=payload.encoding,
            frame_index=payload.frame_index,
            width=width,
            height=height,
            source=f"chunk:{payload.encoding}",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to preprocess video chunk: {exc}") from exc

    return VideoFramePreprocessResult(
        car_id=payload.car_id,
        stream_id=payload.stream_id,
        frame=encode_jpeg_payload(frame.image),
        metadata=frame.metadata,
    )


@app.get("/api/algorithms", response_model=list[AlgorithmInfo])
async def list_algorithms() -> list[AlgorithmInfo]:
    return algorithm_service.catalog.list()


@app.post("/api/algorithms/reload", response_model=list[AlgorithmInfo])
async def reload_algorithms() -> list[AlgorithmInfo]:
    return algorithm_service.catalog.reload()


@app.post("/api/algorithms/{algorithm_id}/run", response_model=AlgorithmRunResult)
async def run_algorithm(algorithm_id: str, payload: AlgorithmRunRequest) -> AlgorithmRunResult:
    if payload.image is None:
        raise HTTPException(status_code=400, detail="image is required")
    try:
        image = decode_jpeg(payload.image)
        return algorithm_service.run_image(
            algorithm_id=algorithm_id,
            image=image,
            car_id=payload.car_id,
            stream_id=payload.stream_id,
            parameters=payload.parameters,
            include_image=payload.include_image,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"algorithm run failed: {exc}") from exc


@app.post("/api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/run-once", response_model=AlgorithmRunResult)
async def run_stream_algorithm_once(
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    include_image: bool = False,
) -> AlgorithmRunResult:
    frame = _read_registered_stream_frame(car_id, stream_id)
    try:
        return algorithm_service.run_image(
            algorithm_id=algorithm_id,
            image=frame.image,
            car_id=car_id,
            stream_id=stream_id,
            parameters={"preprocess": frame.metadata.model_dump(mode="json")},
            include_image=include_image,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"algorithm run failed: {exc}") from exc


@app.post("/api/video/streams/{car_id}/{stream_id}/start", response_model=VideoStreamStatus)
async def start_video_stream_worker(car_id: str, stream_id: str) -> VideoStreamStatus:
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")

    key = (car_id, stream_id)
    worker = stream_workers.get(key)
    if worker is None or worker.done():
        video_streams.mark_started(car_id, stream_id)
        stream_workers[key] = asyncio.create_task(_stream_worker(car_id, stream_id))
    return video_streams.status(car_id, stream_id)


@app.post("/api/video/streams/{car_id}/{stream_id}/stop", response_model=VideoStreamStatus)
async def stop_video_stream_worker(car_id: str, stream_id: str) -> VideoStreamStatus:
    key = (car_id, stream_id)
    worker = stream_workers.pop(key, None)
    if worker is not None:
        worker.cancel()
    if video_streams.get(car_id, stream_id) is None:
        raise HTTPException(status_code=404, detail="video stream not registered")
    video_streams.mark_stopped(car_id, stream_id)
    return video_streams.status(car_id, stream_id)


async def _stream_worker(car_id: str, stream_id: str) -> None:
    try:
        while True:
            runtime = video_streams.get(car_id, stream_id)
            if runtime is None or not runtime.config.enabled:
                return
            config = runtime.config
            try:
                frame = await asyncio.to_thread(
                    open_stream_frame,
                    config.url,
                    width=config.width,
                    height=config.height,
                    timeout_ms=settings.video_capture_timeout_ms,
                    source=f"{config.transport}:{config.stream_id}",
                )
                video_streams.mark_frame(car_id, stream_id)
                frame_event = {
                    "type": "video_frame",
                    "car_id": car_id,
                    "stream_id": stream_id,
                    "metadata": frame.metadata.model_dump(mode="json"),
                }
                await manager.publish(car_id, frame_event)
                algorithm_ids = config.metadata.get("algorithms", [])
                for algorithm_id in algorithm_ids:
                    result = await asyncio.to_thread(
                        algorithm_service.run_image,
                        algorithm_id=algorithm_id,
                        image=frame.image,
                        car_id=car_id,
                        stream_id=stream_id,
                        parameters={"preprocess": frame.metadata.model_dump(mode="json")},
                        include_image=False,
                    )
                    await manager.publish(car_id, result.model_dump(mode="json"))
            except Exception as exc:
                video_streams.mark_error(car_id, stream_id, str(exc))
            await asyncio.sleep(max(0.033, config.sample_interval_ms / 1000.0))
    except asyncio.CancelledError:
        raise
    finally:
        if video_streams.get(car_id, stream_id) is not None:
            video_streams.mark_stopped(car_id, stream_id)


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
