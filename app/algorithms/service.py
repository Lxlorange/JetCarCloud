from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.algorithms.catalog import AlgorithmCatalog
from app.algorithms.local import LocalAlgorithmRunner
from app.algorithms.runner import DockerAlgorithmRunner
from app.schemas import AlgorithmRunResult
from app.video import encode_jpeg_payload


class AlgorithmService:
    def __init__(
        self,
        *,
        catalog: AlgorithmCatalog,
        work_dir: str | Path,
        docker_executable: str = "docker",
    ) -> None:
        self._catalog = catalog
        self._work_dir = Path(work_dir)
        self._project_root = Path(__file__).resolve().parents[2]
        self._docker_runner = DockerAlgorithmRunner(docker_executable=docker_executable)
        self._local_runner = LocalAlgorithmRunner(project_root=self._project_root)

    @property
    def catalog(self) -> AlgorithmCatalog:
        return self._catalog

    def run_image(
        self,
        *,
        algorithm_id: str,
        image: np.ndarray,
        car_id: str,
        stream_id: str,
        parameters: dict | None = None,
        include_image: bool = False,
    ) -> AlgorithmRunResult:
        parameters = dict(parameters or {})
        persist_outputs = bool(parameters.pop("persist_outputs", True))
        spec = self._catalog.require(algorithm_id)
        run_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        run_dir = self._work_dir / algorithm_id / run_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        if persist_outputs:
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

        request = {
            "algorithm_id": algorithm_id,
            "car_id": car_id,
            "stream_id": stream_id,
            "parameters": parameters,
            "input_dir": "/app/data/input",
            "output_dir": "/app/data/output",
        }
        if persist_outputs:
            _write_frame(input_dir / "frame.jpg", image)
            (input_dir / "request.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        started = time.perf_counter()
        result_json: dict
        annotated = None
        error = ""
        ok = True
        runner_outputs: dict
        if spec.runner == "local":
            try:
                result_json, annotated = self._local_runner.run(
                    algorithm_id=algorithm_id,
                    spec=spec,
                    image=image,
                    car_id=car_id,
                    stream_id=stream_id,
                    parameters=parameters,
                )
                if persist_outputs:
                    _write_json(output_dir / "result.json", result_json)
                    if annotated is not None:
                        _write_frame(output_dir / "annotated.jpg", annotated)
                runner_outputs = {
                    "runner": "local",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                }
            except Exception as exc:
                ok = False
                error = str(exc)
                result_json = {
                    "algorithm_id": algorithm_id,
                    "car_id": car_id,
                    "stream_id": stream_id,
                    "error": str(exc),
                }
                if persist_outputs:
                    _write_json(output_dir / "result.json", result_json)
                runner_outputs = {
                    "runner": "local",
                    "returncode": 1,
                    "stdout": "",
                    "stderr": str(exc),
                }
        elif spec.runner == "docker":
            completed = self._docker_runner.run(spec, input_dir=input_dir, output_dir=output_dir)
            result_json = _read_json(output_dir / "result.json")
            annotated_path = output_dir / "annotated.jpg"
            if annotated_path.exists():
                annotated = cv2.imread(str(annotated_path))
            ok = completed.returncode == 0
            if not ok:
                error = completed.stderr.strip() or completed.stdout.strip() or f"docker exited with {completed.returncode}"
            runner_outputs = {
                "runner": "docker",
                "docker_command": getattr(completed, "command", completed.args),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "returncode": completed.returncode,
            }
        else:
            raise ValueError(f"unsupported algorithm runner: {spec.runner}")
        latency_ms = (time.perf_counter() - started) * 1000.0

        annotated_image = None
        if include_image and annotated is not None:
            annotated_image = encode_jpeg_payload(annotated)

        input_files = _list_files(input_dir) if persist_outputs else []
        output_files = _list_files(output_dir) if persist_outputs else []
        outputs = {
            "run_dir": str(run_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "persist_outputs": persist_outputs,
            "input_frame_shape": [int(item) for item in image.shape],
            "input_files": input_files,
            "output_files": output_files,
            "missing_outputs": [],
        }
        outputs.update(runner_outputs)
        for name in spec.outputs:
            path = output_dir / name
            if persist_outputs and path.exists():
                outputs[name] = str(path)
            elif persist_outputs:
                outputs[name] = ""
                outputs["missing_outputs"].append(name)
            else:
                outputs[name] = ""

        return AlgorithmRunResult(
            ok=ok,
            algorithm_id=algorithm_id,
            car_id=car_id,
            stream_id=stream_id,
            runner=spec.runner,
            latency_ms=round(latency_ms, 3),
            result=result_json,
            outputs=outputs,
            annotated_image=annotated_image,
            error=error,
        )


def _write_frame(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"failed to write algorithm input frame: {path}")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_files(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        try:
            size = item.stat().st_size
        except OSError:
            size = 0
        items.append(
            {
                "path": str(item.relative_to(path)),
                "size": int(size),
            }
        )
    return items
