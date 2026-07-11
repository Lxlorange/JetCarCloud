from __future__ import annotations

import asyncio
import time

import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.ai_tasks import AiTaskRegistry, run_yolo_task
from app.config import get_settings
from app.connection_manager import ConnectionManager
from app.features.manhole import ManholeSettings, build_manhole_provider
from app.features.road_defect import RoadDefectSettings, build_road_defect_provider
from app.features.road_inspection import inspect_road
from app.image_codec import decode_jpeg
from app.inference.detector import build_detector
from app.schemas import (
    AiTaskResult,
    AiTaskSpec,
    EdgeFrame,
    FeatureImageUpload,
    ImageUpload,
    InferenceResult,
    ManholeDetectionResult,
    ReferenceUploadResult,
    RoadDefectDetectionResult,
    RoadInspectionResult,
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
manhole_provider = build_manhole_provider(
    ManholeSettings(
        provider=settings.manhole_provider,
        model_path=settings.manhole_model_path,
        backend=settings.manhole_backend,
        yolov5_repo_path=settings.manhole_yolov5_repo_path,
        device=settings.manhole_device,
        confidence=settings.manhole_confidence,
        image_size=settings.manhole_image_size,
        positive_labels={
            label.strip().lower()
            for label in settings.manhole_positive_labels.split(",")
            if label.strip()
        },
        roboflow_api_key=settings.roboflow_api_key,
        roboflow_model_id=settings.roboflow_model_id,
        roboflow_model_version=settings.roboflow_model_version,
        roboflow_api_url=settings.roboflow_api_url,
    )
)
road_defect_provider = build_road_defect_provider(
    RoadDefectSettings(
        model_path=settings.road_defect_model_path,
        backend=settings.road_defect_backend,
        device=settings.road_defect_device,
        confidence=settings.road_defect_confidence,
        image_size=settings.road_defect_image_size,
        positive_labels={
            label.strip().lower()
            for label in settings.road_defect_positive_labels.split(",")
            if label.strip()
        },
    )
)

app = FastAPI(title=settings.app_name)
reference_images = {}
video_streams = VideoStreamRegistry()
ai_tasks = AiTaskRegistry()
stream_workers: dict[tuple[str, str], asyncio.Task] = {}
ai_tasks.register(
    AiTaskSpec(
        task_id="yolo_detection",
        kind="yolo",
        model_path=settings.yolo_model_path,
        backend=settings.yolo_backend,
        metadata={"detector": detector.__class__.__name__},
    )
)
ai_tasks.register(
    AiTaskSpec(
        task_id="road_defect_detection",
        kind="custom",
        model_path=settings.road_defect_model_path,
        backend=settings.road_defect_backend,
        metadata={"provider": road_defect_provider.name},
    )
)
ai_tasks.register(
    AiTaskSpec(
        task_id="road_inspection",
        kind="custom",
        metadata={"features": ["manhole_detection", "road_defect_detection"]},
    )
)
ai_tasks.register(
    AiTaskSpec(
        task_id="manhole_detection",
        kind="custom",
        model_path=settings.manhole_model_path,
        backend=settings.manhole_backend,
        metadata={"provider": manhole_provider.name},
    )
)


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
        "ai_tasks": [task.task_id for task in ai_tasks.list()],
        "manhole_provider": manhole_provider.name,
        "road_defect_provider": road_defect_provider.name,
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


@app.post("/api/video/streams/{car_id}/{stream_id}/tasks/{task_id}/run-once", response_model=AiTaskResult)
async def run_stream_task_once(car_id: str, stream_id: str, task_id: str) -> AiTaskResult:
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")

    task = ai_tasks.get(task_id)
    if task is None or not task.enabled:
        raise HTTPException(status_code=404, detail="ai task not registered or disabled")
    if task.kind != "yolo":
        raise HTTPException(status_code=501, detail="only yolo tasks are executable in-process for now")

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
        raise HTTPException(status_code=502, detail=f"failed to read stream frame: {exc}") from exc

    return run_yolo_task(
        task_id=task.task_id,
        detector=detector,
        image=frame.image,
        car_id=car_id,
        stream_id=stream_id,
        metadata={"preprocess": frame.metadata.model_dump(mode="json")},
    )


@app.post("/api/ai/tasks", response_model=AiTaskSpec)
async def register_ai_task(payload: AiTaskSpec) -> AiTaskSpec:
    return ai_tasks.register(payload)


@app.get("/api/ai/tasks", response_model=list[AiTaskSpec])
async def list_ai_tasks() -> list[AiTaskSpec]:
    return ai_tasks.list()


@app.post("/api/features/manhole/detect", response_model=ManholeDetectionResult)
async def detect_manhole_image(payload: FeatureImageUpload) -> ManholeDetectionResult:
    try:
        image = decode_jpeg(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc

    return manhole_provider.detect(
        image,
        car_id=payload.car_id,
        stream_id="upload",
        metadata={
            "source": "upload",
            "image": {
                "width": payload.image.width,
                "height": payload.image.height,
                "encoding": payload.image.encoding,
            },
        },
        include_image=payload.include_image,
    )


@app.post("/api/video/streams/{car_id}/{stream_id}/features/manhole/run-once", response_model=ManholeDetectionResult)
async def detect_manhole_stream_once(car_id: str, stream_id: str, include_image: bool = False) -> ManholeDetectionResult:
    frame = _read_registered_stream_frame(car_id, stream_id)

    return manhole_provider.detect(
        frame.image,
        car_id=car_id,
        stream_id=stream_id,
        metadata={"preprocess": frame.metadata.model_dump(mode="json")},
        include_image=include_image,
    )


@app.post("/api/features/road-defect/detect", response_model=RoadDefectDetectionResult)
async def detect_road_defect_image(payload: FeatureImageUpload) -> RoadDefectDetectionResult:
    try:
        image = decode_jpeg(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc

    return road_defect_provider.detect(
        image,
        car_id=payload.car_id,
        stream_id="upload",
        metadata={
            "source": "upload",
            "image": {
                "width": payload.image.width,
                "height": payload.image.height,
                "encoding": payload.image.encoding,
            },
        },
        include_image=payload.include_image,
    )


@app.post("/api/video/streams/{car_id}/{stream_id}/features/road-defect/run-once", response_model=RoadDefectDetectionResult)
async def detect_road_defect_stream_once(
    car_id: str,
    stream_id: str,
    include_image: bool = False,
) -> RoadDefectDetectionResult:
    frame = _read_registered_stream_frame(car_id, stream_id)

    return road_defect_provider.detect(
        frame.image,
        car_id=car_id,
        stream_id=stream_id,
        metadata={"preprocess": frame.metadata.model_dump(mode="json")},
        include_image=include_image,
    )


@app.post("/api/features/road-inspection/detect", response_model=RoadInspectionResult)
async def inspect_road_image(payload: FeatureImageUpload) -> RoadInspectionResult:
    try:
        image = decode_jpeg(payload.image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image payload: {exc}") from exc

    return inspect_road(
        image,
        car_id=payload.car_id,
        stream_id="upload",
        manhole_provider=manhole_provider,
        road_defect_provider=road_defect_provider,
        metadata={
            "source": "upload",
            "image": {
                "width": payload.image.width,
                "height": payload.image.height,
                "encoding": payload.image.encoding,
            },
        },
        include_image=payload.include_image,
    )


@app.post("/api/video/streams/{car_id}/{stream_id}/features/road-inspection/run-once", response_model=RoadInspectionResult)
async def inspect_road_stream_once(car_id: str, stream_id: str, include_image: bool = False) -> RoadInspectionResult:
    frame = _read_registered_stream_frame(car_id, stream_id)

    return inspect_road(
        frame.image,
        car_id=car_id,
        stream_id=stream_id,
        manhole_provider=manhole_provider,
        road_defect_provider=road_defect_provider,
        metadata={"preprocess": frame.metadata.model_dump(mode="json")},
        include_image=include_image,
    )


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
                result = run_yolo_task(
                    task_id="yolo_detection",
                    detector=detector,
                    image=frame.image,
                    car_id=car_id,
                    stream_id=stream_id,
                    metadata={"preprocess": frame.metadata.model_dump(mode="json")},
                )
                await manager.publish(car_id, result.model_dump(mode="json"))
                manhole_result = manhole_provider.detect(
                    frame.image,
                    car_id=car_id,
                    stream_id=stream_id,
                    metadata={"preprocess": frame.metadata.model_dump(mode="json")},
                )
                await manager.publish(car_id, manhole_result.model_dump(mode="json"))
                inspection_result = inspect_road(
                    frame.image,
                    car_id=car_id,
                    stream_id=stream_id,
                    manhole_provider=manhole_provider,
                    road_defect_provider=road_defect_provider,
                    metadata={"preprocess": frame.metadata.model_dump(mode="json")},
                    include_image=False,
                )
                await manager.publish(car_id, inspection_result.model_dump(mode="json"))
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
