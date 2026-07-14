from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import requests
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import ValidationError

from app.algorithms import AlgorithmCatalog, AlgorithmService
from app.config import get_settings
from app.connection_manager import ConnectionManager
from app.dashboard import DASHBOARD_HTML
from app.discovery import broadcast_discovery_beacon
from app.image_codec import decode_jpeg
from app.algorithms.local.similarity import extract_similarity_feature, save_similarity_feature
from app.inference.detector import build_detector
from app.schemas import (
    AlgorithmInfo,
    AlgorithmRunRequest,
    AlgorithmRunResult,
    EdgeFrame,
    EdgeControlProxyRequest,
    EdgeEventReport,
    ImageUpload,
    InferenceResult,
    ReferenceUploadResult,
    SimilarityResult,
    SimilaritySearchStartRequest,
    SimilaritySearchStopRequest,
    TaskReportRequest,
    VideoChunkUpload,
    VideoFrameUploadResult,
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
    preprocess_frame,
)

settings = get_settings()
logger = logging.getLogger("jetcar.cloud")
logger.setLevel(logging.INFO)
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
similarity_sessions: dict[tuple[str, str, str], dict] = {}
edge_task_events: dict[tuple[str, str], list[dict]] = {}
video_streams = VideoStreamRegistry()
stream_workers: dict[tuple[str, str], asyncio.Task] = {}
discovery_task: asyncio.Task | None = None


@dataclass
class ProcessedFrameCacheItem:
    image: bytes
    result: AlgorithmRunResult
    timestamp: float


processed_frames: dict[tuple[str, str, str], ProcessedFrameCacheItem] = {}
processing_tasks: set[asyncio.Task] = set()
processing_tasks_by_key: dict[tuple[str, str, str], asyncio.Task] = {}
pending_algorithm_frames: dict[tuple[str, str, str], dict] = {}
pending_retry_tasks: set[tuple[str, str, str]] = set()
algorithm_last_started_at: dict[tuple[str, str, str], float] = {}
debug_dump_dir = Path(settings.debug_dump_dir)
reports_dir = Path(settings.reports_dir)
map_dir = Path(settings.map_dir)


@app.on_event("startup")
async def startup() -> None:
    global discovery_task
    if not settings.discovery_beacon_enabled:
        return
    host = settings.discovery_beacon_host.strip() or _local_lan_ip()
    discovery_task = asyncio.create_task(
        broadcast_discovery_beacon(
            service=settings.app_name,
            host=host,
            port=settings.discovery_beacon_port,
            http_port=settings.port,
            interval_seconds=settings.discovery_beacon_interval_seconds,
        )
    )
    logger.info(
        "cloud discovery beacon started host=%s http_port=%s udp_port=%s",
        host,
        settings.port,
        settings.discovery_beacon_port,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    if discovery_task is not None:
        discovery_task.cancel()


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
        "discovery": {
            "enabled": settings.discovery_beacon_enabled,
            "port": settings.discovery_beacon_port,
            "host": settings.discovery_beacon_host or _local_lan_ip(),
        },
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/unicorn", response_class=HTMLResponse)
async def unicorn_dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/dashboard/state")
async def dashboard_state() -> dict:
    now = time.time()
    algorithms = algorithm_service.catalog.list()
    streams = video_streams.list()
    active_tasks = [
        {
            "car_id": car_id,
            "stream_id": stream_id,
            "algorithm_id": algorithm_id,
        }
        for (car_id, stream_id, algorithm_id), task in processing_tasks_by_key.items()
        if not task.done()
    ]
    return {
        "ok": True,
        "service": settings.app_name,
        "timestamp": now,
        "detector": {
            "class": detector.__class__.__name__,
            "reason": getattr(detector, "reason", ""),
        },
        "connections": await manager.stats(),
        "streams": [item.model_dump(mode="json") for item in streams],
        "algorithms": [item.model_dump(mode="json") for item in algorithms],
        "processing": {
            "active_count": len(active_tasks),
            "active_tasks": active_tasks,
            "pending_task_count": sum(1 for task in processing_tasks if not task.done()),
            "max_concurrent_tasks": settings.algorithm_max_concurrent_tasks,
            "algorithm_min_interval_ms": settings.algorithm_min_interval_ms,
            "video_push_min_interval_ms": settings.video_push_min_interval_ms,
        },
        "similarity_sessions": list(similarity_sessions.values()),
        "edge_tasks": _dashboard_edge_tasks(),
        "processed_frames": _dashboard_processed_frames(now),
        "debug": _dashboard_debug_summary(),
        "discovery": {
            "enabled": settings.discovery_beacon_enabled,
            "port": settings.discovery_beacon_port,
            "host": settings.discovery_beacon_host or _local_lan_ip(),
        },
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


@app.post("/api/similarity/search/start")
async def start_similarity_search(payload: SimilaritySearchStartRequest) -> dict:
    image = decode_jpeg(payload.image)
    target_feature = await asyncio.to_thread(extract_similarity_feature, image)
    session_dir = (
        Path(settings.algorithm_work_dir)
        / "similarity_sessions"
        / _safe_path_part(payload.car_id)
        / _safe_path_part(payload.stream_id)
        / _safe_path_part(payload.algorithm_id)
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    template_path = session_dir / "target.jpg"
    feature_path = session_dir / "target_feature.npy"
    _write_debug_image(template_path, image)
    await asyncio.to_thread(save_similarity_feature, feature_path, target_feature)
    session = {
        "car_id": payload.car_id,
        "stream_id": payload.stream_id,
        "algorithm_id": payload.algorithm_id,
        "threshold": payload.threshold,
        "template_path": str(template_path),
        "feature_path": str(feature_path),
        "feature_dim": int(target_feature.shape[0]),
        "started_at": time.time(),
        "active": True,
    }
    similarity_sessions[(payload.car_id, payload.stream_id, payload.algorithm_id)] = session
    logger.info(
        "similarity search started car_id=%s stream_id=%s algorithm_id=%s template=%s feature=%s dim=%s threshold=%.3f",
        payload.car_id,
        payload.stream_id,
        payload.algorithm_id,
        template_path,
        feature_path,
        target_feature.shape[0],
        payload.threshold,
    )
    return {
        "ok": True,
        "type": "similarity_search_session",
        **session,
        "edge_mask": "similarity",
        "edge_algorithm_ids": [payload.algorithm_id],
    }


@app.post("/api/similarity/search/stop")
async def stop_similarity_search(payload: SimilaritySearchStopRequest) -> dict:
    key = (payload.car_id, payload.stream_id, payload.algorithm_id)
    removed = similarity_sessions.pop(key, None)
    logger.info(
        "similarity search stopped car_id=%s stream_id=%s algorithm_id=%s active_before=%s",
        payload.car_id,
        payload.stream_id,
        payload.algorithm_id,
        removed is not None,
    )
    return {
        "ok": True,
        "type": "similarity_search_stopped",
        "car_id": payload.car_id,
        "stream_id": payload.stream_id,
        "algorithm_id": payload.algorithm_id,
        "active_before": removed is not None,
        "edge_mask": "FF",
        "edge_algorithm_ids": [],
    }


@app.post("/api/debug/edge-control")
async def proxy_edge_control(payload: EdgeControlProxyRequest) -> dict:
    command = dict(payload.command)
    if not command:
        raise HTTPException(status_code=400, detail="command is required")
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            _send_edge_control_command,
            payload.edge_host,
            payload.edge_port,
            command,
            payload.timeout_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"failed to send edge control command: {exc}") from exc
    return {
        "ok": True,
        "edge_host": payload.edge_host,
        "edge_port": payload.edge_port,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "command": command,
        "response": response,
    }


@app.post("/api/edge/events")
async def report_edge_event(payload: EdgeEventReport) -> dict:
    data = dict(payload.payload)
    data.setdefault("type", "edge_similarity_search")
    data["car_id"] = payload.car_id
    data["stream_id"] = payload.stream_id
    data["event"] = payload.event
    if data.get("type") == "edge_road_inspection":
        count = data.get("count", "-")
        target_count = data.get("target_count", "-")
        if payload.event == "inspection_complete":
            data.setdefault("status", "complete")
            data.setdefault("message", f"road inspection complete {count}/{target_count}")
        elif payload.event == "inspection_detection":
            data.setdefault("status", "running")
            data.setdefault("message", f"road inspection detection {count}/{target_count}")
        elif payload.event == "inspection_warning":
            data.setdefault("status", "warning")
            data.setdefault("message", f"road inspection warning: {data.get('reason', '')}")
    await manager.publish(payload.car_id, data)
    if (
        payload.event == "task_status"
        or data.get("type") == "edge_task_state"
        or data.get("type") == "edge_road_inspection"
    ):
        _record_edge_task_event(payload.car_id, payload.stream_id, data)
    logger.info(
        "edge event reported car_id=%s stream_id=%s event=%s",
        payload.car_id,
        payload.stream_id,
        payload.event,
    )
    return {"ok": True, "published": True, "event": payload.event}


@app.get("/api/tasks/{car_id}/{stream_id}/latest")
async def latest_task_events(car_id: str, stream_id: str, limit: int = Query(default=20, ge=1, le=200)) -> dict:
    events = edge_task_events.get((car_id, stream_id), [])
    return {
        "ok": True,
        "car_id": car_id,
        "stream_id": stream_id,
        "events": events[-limit:],
        "latest": events[-1] if events else None,
    }


@app.post("/api/tasks/report")
async def write_task_report(payload: TaskReportRequest) -> dict:
    task_id = payload.task_id.strip() or f"{payload.mode or 'task'}-{int(time.time() * 1000)}"
    report_dir = reports_dir / _safe_path_part(payload.car_id) / _safe_path_part(payload.stream_id) / _safe_path_part(task_id)
    image_dir = report_dir / "images"
    report_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key, item in processed_frames.items():
        if key[0] != payload.car_id or key[1] != payload.stream_id:
            continue
        algorithm_id = key[2]
        image_name = ""
        if item.result.annotated_image is not None:
            image_name = f"{_safe_path_part(algorithm_id)}-{int(item.timestamp * 1000)}.jpg"
            try:
                (image_dir / image_name).write_bytes(base64.b64decode(item.result.annotated_image.data))
            except Exception:
                logger.exception("failed to save report image algorithm_id=%s", algorithm_id)
                image_name = ""
        results.append(
            {
                "key": "/".join(key),
                "algorithm_id": algorithm_id,
                "timestamp": item.timestamp,
                "image": f"images/{image_name}" if image_name else "",
                "result": item.result.model_dump(mode="json"),
            }
        )
    results.sort(key=lambda item: item["timestamp"], reverse=True)
    report = {
        "ok": True,
        "car_id": payload.car_id,
        "stream_id": payload.stream_id,
        "task_id": task_id,
        "mode": payload.mode,
        "summary": payload.summary,
        "result_count": len(results),
        "results": results,
        "created_at": time.time(),
    }
    path = report_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path = report_dir / "index.html"
    html_path.write_text(_render_task_report_html(report), encoding="utf-8")
    return {
        "ok": True,
        "task_id": task_id,
        "report_path": str(path),
        "html_path": str(html_path),
        "result_count": len(results),
        "report_url": f"/api/tasks/reports/{payload.car_id}/{payload.stream_id}/{task_id}/index.html",
    }


@app.get("/api/tasks/reports")
async def list_task_reports(car_id: str | None = None, stream_id: str | None = None) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pattern = "*/*/report.json"
    for path in reports_dir.glob(pattern):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if car_id and data.get("car_id") != car_id:
            continue
        if stream_id and data.get("stream_id") != stream_id:
            continue
        rows.append(
            {
                "car_id": data.get("car_id", ""),
                "stream_id": data.get("stream_id", ""),
                "task_id": data.get("task_id", ""),
                "mode": data.get("mode", ""),
                "created_at": data.get("created_at", 0),
                "result_count": data.get("result_count", 0),
                "report_url": f"/api/tasks/reports/{data.get('car_id', '')}/{data.get('stream_id', '')}/{data.get('task_id', '')}/index.html",
            }
        )
    rows.sort(key=lambda item: item["created_at"], reverse=True)
    return {"ok": True, "reports": rows}


@app.get("/api/tasks/reports/{car_id}/{stream_id}/{task_id}/{path:path}")
async def get_task_report_file(car_id: str, stream_id: str, task_id: str, path: str) -> FileResponse:
    root = (
        reports_dir
        / _safe_path_part(car_id)
        / _safe_path_part(stream_id)
        / _safe_path_part(task_id)
    ).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="report file not found")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="report file not found")
    return FileResponse(target)


@app.get("/api/maps")
async def list_maps() -> dict:
    map_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for yaml_path in sorted(map_dir.glob("*.yaml")):
        items.append({"map_id": yaml_path.stem, "yaml_path": str(yaml_path), "metadata_url": f"/api/maps/{yaml_path.stem}"})
    return {"ok": True, "maps": items}


@app.get("/api/maps/{map_id}")
async def get_map_metadata(map_id: str) -> dict:
    yaml_path = map_dir / f"{_safe_path_part(map_id)}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="map yaml not found")
    metadata = _read_map_yaml(yaml_path)
    image_name = str(metadata.get("image") or "")
    image_path = (yaml_path.parent / image_name).resolve() if image_name else None
    return {
        "ok": True,
        "map_id": map_id,
        "yaml_path": str(yaml_path),
        "metadata": metadata,
        "image_url": f"/api/maps/{map_id}/image" if image_path and image_path.exists() else "",
    }


@app.get("/api/maps/{map_id}/image")
async def get_map_image(map_id: str) -> FileResponse:
    yaml_path = map_dir / f"{_safe_path_part(map_id)}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="map yaml not found")
    metadata = _read_map_yaml(yaml_path)
    image_name = str(metadata.get("image") or "")
    if not image_name:
        raise HTTPException(status_code=404, detail="map image not configured")
    root = yaml_path.parent.resolve()
    image_path = (root / image_name).resolve()
    try:
        image_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="map image not found")
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="map image not found")
    return FileResponse(image_path)


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


def _send_edge_control_command(host: str, port: int, command: dict, timeout_seconds: float) -> dict:
    message = json.dumps(command, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((host, int(port)), timeout=timeout_seconds) as sock:
        sock.settimeout(timeout_seconds)
        sock.sendall(message.encode("utf-8"))
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not raw:
        return {"raw": ""}
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    if isinstance(loaded, dict):
        return loaded
    return {"raw": loaded}


def _read_registered_stream_frame(car_id: str, stream_id: str):
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        raise HTTPException(status_code=404, detail="video stream not registered")

    config = runtime.config
    if config.transport == "push":
        frame = video_streams.latest_frame(car_id, stream_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="no pushed frame received yet")
        return frame

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


@app.post("/api/video/streams/{car_id}/{stream_id}/frames", response_model=VideoFrameUploadResult)
async def push_video_frame(
    car_id: str,
    stream_id: str,
    payload: ImageUpload,
    algorithm_id: str | None = None,
    algorithm_ids: str | None = None,
    include_image: bool = True,
) -> VideoFrameUploadResult:
    return await _accept_pushed_frame(
        payload=payload,
        car_id=car_id,
        stream_id=stream_id,
        algorithm_id=algorithm_id,
        algorithm_ids=algorithm_ids,
        include_image=include_image,
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
        result = algorithm_service.run_image(
            algorithm_id=algorithm_id,
            image=image,
            car_id=payload.car_id,
            stream_id=payload.stream_id,
            parameters=payload.parameters,
            include_image=payload.include_image,
        )
        debug_dir = _debug_dump_algorithm_result(
            car_id=payload.car_id,
            stream_id=payload.stream_id,
            algorithm_id=algorithm_id,
            result=result,
        )
        logger.info(
            "algorithm upload result car_id=%s stream_id=%s algorithm_id=%s ok=%s latency_ms=%.3f detections=%s annotated=%s debug_dir=%s error=%s",
            payload.car_id,
            payload.stream_id,
            algorithm_id,
            result.ok,
            result.latency_ms,
            _summarize_detection_count(result.result),
            result.annotated_image is not None,
            debug_dir or "",
            result.error,
        )
        return result
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
    parameters = {"preprocess": frame.metadata.model_dump(mode="json")}
    if algorithm_id == "yolov5-similarity":
        session = similarity_sessions.get((car_id, stream_id, algorithm_id))
        if session is not None:
            parameters.update(
                {
                    "template_path": session["template_path"],
                    "feature_path": session.get("feature_path", ""),
                    "threshold": session["threshold"],
                }
            )
    try:
        result = algorithm_service.run_image(
            algorithm_id=algorithm_id,
            image=frame.image,
            car_id=car_id,
            stream_id=stream_id,
            parameters=parameters,
            include_image=include_image,
        )
        _cache_processed_frame(car_id, stream_id, algorithm_id, result)
        debug_dir = _debug_dump_algorithm_result(
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            result=result,
        )
        logger.info(
            "algorithm run-once result car_id=%s stream_id=%s algorithm_id=%s ok=%s latency_ms=%.3f detections=%s annotated=%s debug_dir=%s error=%s",
            car_id,
            stream_id,
            algorithm_id,
            result.ok,
            result.latency_ms,
            _summarize_detection_count(result.result),
            result.annotated_image is not None,
            debug_dir or "",
            result.error,
        )
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"algorithm run failed: {exc}") from exc


@app.get("/api/video/streams/{car_id}/{stream_id}/algorithms/{algorithm_id}/mjpeg")
async def stream_algorithm_mjpeg(
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    fps: float = Query(default=1.0, ge=0.1, le=10.0),
    fallback_original: bool = True,
) -> StreamingResponse:
    try:
        algorithm_service.catalog.require(algorithm_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return StreamingResponse(
        _processed_mjpeg_generator(
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            fps=fps,
            fallback_original=fallback_original,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
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
                frame_event = {
                    "type": "video_frame",
                    "car_id": car_id,
                    "stream_id": stream_id,
                    "metadata": frame.metadata.model_dump(mode="json"),
                }
                await manager.publish(car_id, frame_event)
                algorithm_ids = _algorithm_ids_for_push(runtime, None, None)
                queued, skipped = _queue_algorithms_for_frame(
                    frame=frame,
                    car_id=car_id,
                    stream_id=stream_id,
                    algorithm_ids=algorithm_ids,
                    include_image=False,
                )
                logger.info(
                    "stream worker sampled frame car_id=%s stream_id=%s queued=%s skipped=%s",
                    car_id,
                    stream_id,
                    queued,
                    skipped,
                )
            except Exception as exc:
                video_streams.mark_error(car_id, stream_id, str(exc))
            await asyncio.sleep(max(0.033, config.sample_interval_ms / 1000.0))
    except asyncio.CancelledError:
        raise
    finally:
        if video_streams.get(car_id, stream_id) is not None:
            video_streams.mark_stopped(car_id, stream_id)


async def _processed_mjpeg_generator(
    *,
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    fps: float,
    fallback_original: bool,
):
    interval = 1.0 / fps
    last_timestamp = 0.0
    while True:
        started = time.perf_counter()
        try:
            cached = processed_frames.get((car_id, stream_id, algorithm_id))
            jpeg = None
            if cached is not None and cached.timestamp > last_timestamp:
                jpeg = cached.image
                last_timestamp = cached.timestamp
            elif fallback_original:
                runtime = video_streams.get(car_id, stream_id)
                if runtime is not None and runtime.latest_frame is not None:
                    frame = runtime.latest_frame
                    jpeg = _encode_jpeg_bytes(frame.image)
                else:
                    jpeg = _waiting_frame_jpeg(
                        f"waiting for {car_id}/{stream_id}",
                        f"algorithm: {algorithm_id}",
                    )
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if video_streams.get(car_id, stream_id) is not None:
                video_streams.mark_error(car_id, stream_id, str(exc))

        elapsed = time.perf_counter() - started
        await asyncio.sleep(max(0.0, interval - elapsed))


def _algorithm_ids_for_push(runtime, algorithm_id: str | None, algorithm_ids: str | None) -> list[str]:
    requested = _parse_algorithm_ids(algorithm_ids) + _parse_algorithm_ids(algorithm_id)
    if requested:
        return requested
    if runtime is None:
        return []
    configured = runtime.config.metadata.get("algorithms", [])
    if isinstance(configured, str):
        return _parse_algorithm_ids(configured)
    if isinstance(configured, list):
        return [str(item) for item in configured if str(item)]
    return []


def _parse_algorithm_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


async def _accept_pushed_frame(
    *,
    payload: ImageUpload,
    car_id: str,
    stream_id: str,
    algorithm_id: str | None,
    algorithm_ids: str | None,
    include_image: bool,
) -> VideoFrameUploadResult:
    runtime = video_streams.get(car_id, stream_id)
    requested_algorithm_ids = _algorithm_ids_for_push(runtime, algorithm_id, algorithm_ids)
    for item in requested_algorithm_ids:
        try:
            algorithm_service.catalog.require(item)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    throttle_result = _throttled_push_result(
        runtime=runtime,
        car_id=car_id,
        stream_id=stream_id,
        algorithm_ids=requested_algorithm_ids,
    )
    if throttle_result is not None:
        return throttle_result

    image = decode_jpeg(payload.image)
    if runtime is None:
        config = VideoStreamConfig(
            car_id=car_id,
            stream_id=stream_id,
            url=f"push://{car_id}/{stream_id}",
            transport="push",
            width=payload.image.width or settings.video_default_width,
            height=payload.image.height or settings.video_default_height,
            enabled=True,
        )
        video_streams.upsert(config)
    else:
        config = runtime.config
        if config.transport != "push":
            config = config.model_copy(update={"transport": "push", "url": f"push://{car_id}/{stream_id}"})
            video_streams.upsert(config)

    frame = preprocess_frame(
        image,
        width=payload.image.width or settings.video_default_width,
        height=payload.image.height or settings.video_default_height,
        source=f"push:{stream_id}",
        letterbox=False,
    )
    video_streams.store_frame(car_id, stream_id, frame)
    status = video_streams.status(car_id, stream_id)
    queued_algorithm_ids, skipped_algorithm_ids = _queue_algorithms_for_frame(
        frame=frame,
        car_id=car_id,
        stream_id=stream_id,
        algorithm_ids=requested_algorithm_ids,
        include_image=include_image,
    )
    debug_frame_dir = _debug_dump_received_frame(
        car_id=car_id,
        stream_id=stream_id,
        frame_count=status.frame_count,
        image=image,
        frame=frame,
        payload=payload,
        requested_algorithm_ids=requested_algorithm_ids,
        queued_algorithm_ids=queued_algorithm_ids,
        skipped_algorithm_ids=skipped_algorithm_ids,
    )
    logger.info(
        "received frame car_id=%s stream_id=%s frame_count=%s image=%sx%s processed=%sx%s queued=%s skipped=%s debug_dir=%s",
        car_id,
        stream_id,
        status.frame_count,
        payload.image.width,
        payload.image.height,
        frame.metadata.resized_width,
        frame.metadata.resized_height,
        queued_algorithm_ids,
        skipped_algorithm_ids,
        debug_frame_dir or "",
    )
    await manager.publish(
        car_id,
        {
            "type": "video_frame",
            "car_id": car_id,
            "stream_id": stream_id,
            "metadata": frame.metadata.model_dump(mode="json"),
        },
    )
    return VideoFrameUploadResult(
        car_id=car_id,
        stream_id=stream_id,
        frame_count=status.frame_count,
        metadata=frame.metadata,
        frame_accepted=True,
        algorithms_queued=queued_algorithm_ids,
        algorithms_skipped=skipped_algorithm_ids,
    )


def _queue_algorithms_for_frame(
    *,
    frame,
    car_id: str,
    stream_id: str,
    algorithm_ids: list[str],
    include_image: bool,
) -> tuple[list[str], list[dict]]:
    queued: list[str] = []
    skipped: list[dict] = []
    for algorithm_id in algorithm_ids:
        key = (car_id, stream_id, algorithm_id)
        parameters = {
            "preprocess": frame.metadata.model_dump(mode="json"),
            "persist_outputs": settings.algorithm_realtime_persist_outputs,
        }
        if algorithm_id == "yolov5-similarity":
            session = similarity_sessions.get(key)
            if session is None or not session.get("active"):
                skipped.append({"algorithm_id": algorithm_id, "reason": "similarity_session_not_started"})
                continue
            parameters.update(
                {
                    "template_path": session["template_path"],
                    "feature_path": session.get("feature_path", ""),
                    "threshold": session["threshold"],
                }
            )

        running = processing_tasks_by_key.get(key)
        if running is not None and not running.done():
            pending_algorithm_frames[key] = {
                "frame": frame,
                "car_id": car_id,
                "stream_id": stream_id,
                "algorithm_id": algorithm_id,
                "parameters": parameters,
                "include_image": include_image,
            }
            skipped.append({"algorithm_id": algorithm_id, "reason": "algorithm_busy_latest_frame_kept"})
            continue

        active_tasks = sum(1 for task in processing_tasks if not task.done())
        if active_tasks >= settings.algorithm_max_concurrent_tasks:
            pending_algorithm_frames[key] = {
                "frame": frame,
                "car_id": car_id,
                "stream_id": stream_id,
                "algorithm_id": algorithm_id,
                "parameters": parameters,
                "include_image": include_image,
            }
            _schedule_pending_retry(key, delay_seconds=0.1)
            skipped.append(
                {
                    "algorithm_id": algorithm_id,
                    "reason": "global_algorithm_limit_latest_frame_kept",
                    "active_tasks": active_tasks,
                    "limit": settings.algorithm_max_concurrent_tasks,
                }
            )
            continue

        now = time.monotonic()
        min_interval = settings.algorithm_min_interval_ms / 1000.0
        last_started = algorithm_last_started_at.get(key, 0.0)
        if min_interval > 0 and now - last_started < min_interval:
            retry_after_seconds = max(0.0, min_interval - (now - last_started))
            pending_algorithm_frames[key] = {
                "frame": frame,
                "car_id": car_id,
                "stream_id": stream_id,
                "algorithm_id": algorithm_id,
                "parameters": parameters,
                "include_image": include_image,
            }
            _schedule_pending_retry(key, delay_seconds=retry_after_seconds)
            skipped.append(
                {
                    "algorithm_id": algorithm_id,
                    "reason": "algorithm_rate_limited_latest_frame_kept",
                    "retry_after_ms": int(retry_after_seconds * 1000),
                }
            )
            continue

        _start_algorithm_task(
            key=key,
            frame=frame,
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            parameters=parameters,
            include_image=include_image,
        )
        queued.append(algorithm_id)
    return queued, skipped


def _start_algorithm_task(
    *,
    key: tuple[str, str, str],
    frame,
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    parameters: dict,
    include_image: bool,
) -> None:
    algorithm_last_started_at[key] = time.monotonic()
    task = asyncio.create_task(
        _run_algorithm_for_pushed_frame(
            frame=frame,
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            parameters=parameters,
            include_image=include_image,
        )
    )
    processing_tasks.add(task)
    processing_tasks_by_key[key] = task
    task.add_done_callback(lambda completed, task_key=key: _on_algorithm_task_done(completed, task_key))


def _on_algorithm_task_done(completed: asyncio.Task, task_key: tuple[str, str, str]) -> None:
    processing_tasks.discard(completed)
    if processing_tasks_by_key.get(task_key) is completed:
        processing_tasks_by_key.pop(task_key, None)
    pending = pending_algorithm_frames.pop(task_key, None)
    if pending is None:
        return
    active_tasks = sum(1 for task in processing_tasks if not task.done())
    if active_tasks >= settings.algorithm_max_concurrent_tasks:
        pending_algorithm_frames[task_key] = pending
        _schedule_pending_retry(task_key, delay_seconds=0.1)
        return
    _start_algorithm_task(key=task_key, **pending)


def _schedule_pending_retry(task_key: tuple[str, str, str], *, delay_seconds: float) -> None:
    if task_key in pending_retry_tasks:
        return
    pending_retry_tasks.add(task_key)
    asyncio.create_task(_retry_pending_algorithm(task_key, delay_seconds=delay_seconds))


async def _retry_pending_algorithm(task_key: tuple[str, str, str], *, delay_seconds: float) -> None:
    await asyncio.sleep(delay_seconds)
    pending_retry_tasks.discard(task_key)
    if task_key in processing_tasks_by_key:
        return
    pending = pending_algorithm_frames.pop(task_key, None)
    if pending is None:
        return
    active_tasks = sum(1 for task in processing_tasks if not task.done())
    if active_tasks >= settings.algorithm_max_concurrent_tasks:
        pending_algorithm_frames[task_key] = pending
        _schedule_pending_retry(task_key, delay_seconds=delay_seconds)
        return
    _start_algorithm_task(key=task_key, **pending)


def _throttled_push_result(
    *,
    runtime,
    car_id: str,
    stream_id: str,
    algorithm_ids: list[str],
) -> VideoFrameUploadResult | None:
    if runtime is None or runtime.latest_frame is None or runtime.last_frame_at is None:
        return None

    min_interval = settings.video_push_min_interval_ms / 1000.0
    elapsed = time.time() - runtime.last_frame_at
    if min_interval <= 0 or elapsed >= min_interval:
        return None

    retry_after_ms = int(max(0.0, min_interval - elapsed) * 1000)
    skipped = [
        {
            "algorithm_id": item,
            "reason": "input_rate_limited",
            "retry_after_ms": retry_after_ms,
        }
        for item in algorithm_ids
    ]
    logger.info(
        "dropped pushed frame car_id=%s stream_id=%s reason=input_rate_limited retry_after_ms=%s algorithms=%s",
        car_id,
        stream_id,
        retry_after_ms,
        algorithm_ids,
    )
    return VideoFrameUploadResult(
        car_id=car_id,
        stream_id=stream_id,
        frame_count=runtime.frame_count,
        metadata=runtime.latest_frame.metadata,
        frame_accepted=False,
        algorithms_queued=[],
        algorithms_skipped=skipped,
    )


async def _run_algorithm_for_pushed_frame(
    *,
    frame,
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    parameters: dict,
    include_image: bool,
) -> None:
    try:
        result = await asyncio.to_thread(
            algorithm_service.run_image,
            algorithm_id=algorithm_id,
            image=frame.image,
            car_id=car_id,
            stream_id=stream_id,
            parameters=parameters,
            include_image=include_image,
        )
        _cache_processed_frame(car_id, stream_id, algorithm_id, result)
        debug_dir = _debug_dump_algorithm_result(
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            result=result,
        )
        logger.info(
            "algorithm result car_id=%s stream_id=%s algorithm_id=%s ok=%s latency_ms=%.3f detections=%s annotated=%s debug_dir=%s error=%s",
            car_id,
            stream_id,
            algorithm_id,
            result.ok,
            result.latency_ms,
            _summarize_detection_count(result.result),
            result.annotated_image is not None,
            debug_dir or "",
            result.error,
        )
        await manager.publish(car_id, result.model_dump(mode="json"))
    except Exception as exc:
        video_streams.mark_error(car_id, stream_id, str(exc))
        debug_dir = _debug_dump_algorithm_error(
            car_id=car_id,
            stream_id=stream_id,
            algorithm_id=algorithm_id,
            error=exc,
        )
        logger.exception(
            "algorithm failed car_id=%s stream_id=%s algorithm_id=%s debug_dir=%s",
            car_id,
            stream_id,
            algorithm_id,
            debug_dir or "",
        )
        await manager.publish(
            car_id,
            {
                "type": "algorithm_result",
                "ok": False,
                "algorithm_id": algorithm_id,
                "car_id": car_id,
                "stream_id": stream_id,
                "runner": "local",
                "latency_ms": 0.0,
                "result": {},
                "outputs": {},
                "annotated_image": None,
                "error": str(exc),
            },
        )


def _cache_processed_frame(car_id: str, stream_id: str, algorithm_id: str, result: AlgorithmRunResult) -> None:
    jpeg = _result_jpeg_bytes(result)
    if jpeg is None:
        return
    processed_frames[(car_id, stream_id, algorithm_id)] = ProcessedFrameCacheItem(
        image=jpeg,
        result=result,
        timestamp=time.time(),
    )


def _debug_dump_received_frame(
    *,
    car_id: str,
    stream_id: str,
    frame_count: int,
    image,
    frame,
    payload: ImageUpload,
    requested_algorithm_ids: list[str],
    queued_algorithm_ids: list[str],
    skipped_algorithm_ids: list[dict],
) -> str:
    if not settings.debug_dump_enabled:
        return ""
    if not queued_algorithm_ids and skipped_algorithm_ids and not settings.debug_dump_skipped_frames:
        return ""
    frame_dir = _debug_frame_dir(car_id, stream_id, frame_count)
    frame_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "event": "frame_received",
        "car_id": car_id,
        "stream_id": stream_id,
        "frame_count": frame_count,
        "received_image": {
            "encoding": payload.image.encoding,
            "width": payload.image.width,
            "height": payload.image.height,
            "base64_length": len(payload.image.data),
        },
        "processed_metadata": frame.metadata.model_dump(mode="json"),
        "algorithms_requested": requested_algorithm_ids,
        "algorithms_queued": queued_algorithm_ids,
        "algorithms_skipped": skipped_algorithm_ids,
        "saved_at": time.time(),
    }
    _write_json(frame_dir / "frame_metadata.json", metadata)
    if settings.debug_save_images:
        _write_debug_image(frame_dir / "received.jpg", image)
        _write_debug_image(frame_dir / "preprocessed.jpg", frame.image)
    return str(frame_dir)


def _debug_dump_algorithm_result(
    *,
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    result: AlgorithmRunResult,
) -> str:
    if not settings.debug_dump_enabled:
        return ""
    frame_count = _current_frame_count(car_id, stream_id)
    result_dir = _debug_frame_dir(car_id, stream_id, frame_count) / "algorithms" / _safe_path_part(algorithm_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_payload = result.model_dump(mode="json", exclude={"annotated_image"})
    result_payload["annotated_image_present"] = result.annotated_image is not None
    result_payload["saved_at"] = time.time()
    _write_json(result_dir / "result.json", result_payload)
    _write_json(
        result_dir / "diagnostics.json",
        {
            "docker_command": result.outputs.get("docker_command"),
            "returncode": result.outputs.get("returncode"),
            "run_dir": result.outputs.get("run_dir"),
            "input_dir": result.outputs.get("input_dir"),
            "output_dir": result.outputs.get("output_dir"),
            "input_frame_shape": result.outputs.get("input_frame_shape"),
            "input_files": result.outputs.get("input_files"),
            "output_files": result.outputs.get("output_files"),
            "missing_outputs": result.outputs.get("missing_outputs"),
            "saved_at": time.time(),
        },
    )

    outputs = result.outputs or {}
    stdout = str(outputs.get("stdout", ""))
    stderr = str(outputs.get("stderr", ""))
    if stdout:
        (result_dir / "stdout.txt").write_text(stdout, encoding="utf-8", errors="replace")
    if stderr:
        (result_dir / "stderr.txt").write_text(stderr, encoding="utf-8", errors="replace")

    if settings.debug_save_algorithm_outputs and result.annotated_image is not None:
        try:
            (result_dir / "annotated.jpg").write_bytes(base64.b64decode(result.annotated_image.data))
        except Exception:
            logger.exception("failed to save debug annotated image for %s", algorithm_id)
    if settings.debug_save_algorithm_outputs:
        _copy_algorithm_output_files(outputs, result_dir / "outputs")

    return str(result_dir)


def _debug_dump_algorithm_error(
    *,
    car_id: str,
    stream_id: str,
    algorithm_id: str,
    error: Exception,
) -> str:
    if not settings.debug_dump_enabled:
        return ""
    frame_count = _current_frame_count(car_id, stream_id)
    result_dir = _debug_frame_dir(car_id, stream_id, frame_count) / "algorithms" / _safe_path_part(algorithm_id)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        result_dir / "error.json",
        {
            "event": "algorithm_error",
            "car_id": car_id,
            "stream_id": stream_id,
            "algorithm_id": algorithm_id,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "saved_at": time.time(),
        },
    )
    return str(result_dir)


def _debug_frame_dir(car_id: str, stream_id: str, frame_count: int) -> Path:
    return (
        debug_dump_dir
        / _safe_path_part(car_id)
        / _safe_path_part(stream_id)
        / f"frame_{frame_count:06d}"
    )


def _current_frame_count(car_id: str, stream_id: str) -> int:
    runtime = video_streams.get(car_id, stream_id)
    if runtime is None:
        return 0
    return runtime.frame_count


def _write_debug_image(path: Path, image) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"failed to write debug image: {path}")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_task_report_html(report: dict) -> str:
    rows = []
    for item in report.get("results", []):
        result = item.get("result", {})
        payload = result.get("result", {}) if isinstance(result, dict) else {}
        summary = _dashboard_result_summary(payload if isinstance(payload, dict) else {})
        image = item.get("image") or ""
        image_html = f'<img src="{image}" alt="{item.get("algorithm_id", "")}">' if image else ""
        rows.append(
            "<tr>"
            f"<td>{_html_escape(item.get('algorithm_id', ''))}</td>"
            f"<td>{time.strftime('%H:%M:%S', time.localtime(float(item.get('timestamp', 0) or 0)))}</td>"
            f"<td>{_html_escape(summary)}</td>"
            f"<td>{image_html}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>JetCar Task Report</title>
  <style>
    body {{ font: 14px/1.5 system-ui, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1 {{ font-size: 22px; margin: 0 0 12px; }}
    .meta {{ margin-bottom: 16px; color: #64748b; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #dbe3ef; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 320px; max-height: 220px; object-fit: contain; display: block; }}
  </style>
</head>
<body>
  <h1>JetCar Task Report</h1>
  <div class="meta">
    car={_html_escape(report.get('car_id', ''))}
    stream={_html_escape(report.get('stream_id', ''))}
    task={_html_escape(report.get('task_id', ''))}
    mode={_html_escape(report.get('mode', ''))}
    results={_html_escape(report.get('result_count', 0))}
  </div>
  <table>
    <thead><tr><th>Algorithm</th><th>Time</th><th>Summary</th><th>Image</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def _html_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _read_map_yaml(path: Path) -> dict:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to read map yaml: {exc}") from exc


def _copy_algorithm_output_files(outputs: dict, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for key, value in outputs.items():
        if key in {"stdout", "stderr", "returncode", "run_dir"} or not value:
            continue
        source = Path(str(value))
        if not source.exists() or not source.is_file():
            continue
        target = target_dir / _safe_path_part(source.name)
        try:
            shutil.copy2(source, target)
        except OSError:
            logger.exception("failed to copy algorithm output file source=%s target=%s", source, target)


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value) or "unknown"


def _dashboard_processed_frames(now: float) -> list[dict]:
    rows: list[dict] = []
    for (car_id, stream_id, algorithm_id), item in processed_frames.items():
        result = item.result
        rows.append(
            {
                "car_id": car_id,
                "stream_id": stream_id,
                "algorithm_id": algorithm_id,
                "timestamp": item.timestamp,
                "age_seconds": round(max(0.0, now - item.timestamp), 3),
                "ok": result.ok,
                "runner": result.runner,
                "latency_ms": round(result.latency_ms, 3),
                "error": result.error,
                "detection_count": _summarize_detection_count(result.result),
                "summary": _dashboard_result_summary(result.result),
                "annotated_image_present": result.annotated_image is not None,
                "mjpeg_url": (
                    f"/api/video/streams/{car_id}/{stream_id}/algorithms/"
                    f"{algorithm_id}/mjpeg?fps=2"
                ),
            }
        )
    rows.sort(key=lambda item: item["timestamp"], reverse=True)
    return rows


def _record_edge_task_event(car_id: str, stream_id: str, data: dict) -> None:
    key = (car_id, stream_id)
    event = dict(data)
    event["received_at"] = time.time()
    events = edge_task_events.setdefault(key, [])
    events.append(event)
    if len(events) > 200:
        del events[: len(events) - 200]


def _dashboard_edge_tasks() -> list[dict]:
    rows = []
    for (car_id, stream_id), events in edge_task_events.items():
        latest = events[-1] if events else {}
        rows.append(
            {
                "car_id": car_id,
                "stream_id": stream_id,
                "event_count": len(events),
                "latest": latest,
            }
        )
    rows.sort(key=lambda item: item["latest"].get("received_at", 0), reverse=True)
    return rows


def _dashboard_result_summary(payload: dict) -> str:
    if not payload:
        return ""
    for key in ("summary", "message", "status"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value[:240]
    detections = payload.get("detections")
    if isinstance(detections, list) and detections:
        labels = []
        for item in detections[:5]:
            if isinstance(item, dict):
                label = item.get("label") or item.get("class") or item.get("name")
                if label:
                    labels.append(str(label))
        if labels:
            suffix = "" if len(detections) <= 5 else f" +{len(detections) - 5}"
            return ", ".join(labels) + suffix
    return json.dumps(payload, ensure_ascii=False)[:240]


def _dashboard_debug_summary() -> dict:
    recent_frames: list[dict] = []
    if debug_dump_dir.exists():
        frame_dirs = [
            path
            for path in debug_dump_dir.glob("*/*/frame_*")
            if path.is_dir()
        ]
        frame_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in frame_dirs[:12]:
            algorithm_dir = path / "algorithms"
            algorithms = []
            if algorithm_dir.exists():
                algorithms = sorted(item.name for item in algorithm_dir.iterdir() if item.is_dir())
            recent_frames.append(
                {
                    "car_id": path.parent.parent.name,
                    "stream_id": path.parent.name,
                    "frame": path.name,
                    "modified_at": path.stat().st_mtime,
                    "algorithms": algorithms,
                }
            )
    return {
        "enabled": settings.debug_dump_enabled,
        "dir": str(debug_dump_dir),
        "save_images": settings.debug_save_images,
        "save_algorithm_outputs": settings.debug_save_algorithm_outputs,
        "recent_frames": recent_frames,
    }


def _summarize_detection_count(payload: dict) -> int | str:
    for key in ("detection_count", "detections_count", "count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    detections = payload.get("detections")
    if isinstance(detections, list):
        return len(detections)
    return "unknown"


def _result_jpeg_bytes(result: AlgorithmRunResult) -> bytes | None:
    if result.annotated_image is None:
        return None
    return base64.b64decode(result.annotated_image.data)


def _encode_jpeg_bytes(image) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("failed to encode jpeg frame")
    return encoded.tobytes()


def _waiting_frame_jpeg(line1: str, line2: str = "") -> bytes:
    import numpy as np

    image = np.full((360, 640, 3), 245, dtype=np.uint8)
    cv2.putText(image, "JetCarCloud preview", (32, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(image, line1[:60], (32, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(image, line2[:60], (32, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(image, "no frame received yet", (32, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 110, 180), 2, cv2.LINE_AA)
    return _encode_jpeg_bytes(image)


def _local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


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


@app.websocket("/ws/video/{car_id}/{stream_id}/edge")
async def edge_video_socket(websocket: WebSocket, car_id: str, stream_id: str) -> None:
    await websocket.accept()
    algorithm_id = websocket.query_params.get("algorithm_id")
    algorithm_ids = websocket.query_params.get("algorithm_ids") or websocket.query_params.get("algorithms")
    include_image = websocket.query_params.get("include_image", "true").lower() != "false"
    while True:
        try:
            raw = await websocket.receive_json()
            payload = ImageUpload.model_validate(raw)
            result = await _accept_pushed_frame(
                payload=payload,
                car_id=car_id,
                stream_id=stream_id,
                algorithm_id=algorithm_id,
                algorithm_ids=algorithm_ids,
                include_image=include_image,
            )
            await websocket.send_json(result.model_dump(mode="json"))
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
