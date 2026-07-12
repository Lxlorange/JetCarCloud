from __future__ import annotations

import subprocess
from pathlib import Path

from app.schemas import AlgorithmInfo


class DockerAlgorithmRunner:
    def __init__(self, *, docker_executable: str = "docker") -> None:
        self._docker = docker_executable

    def run(self, spec: AlgorithmInfo, *, input_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
        if not spec.image:
            raise ValueError(f"algorithm {spec.algorithm_id} has no docker image")

        command = [
            self._docker,
            "run",
            "--rm",
            "-v",
            f"{input_dir.resolve()}:/app/data/input",
            "-v",
            f"{output_dir.resolve()}:/app/data/output",
            spec.image,
        ]

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _decode_output(exc.output)
            stderr = _decode_output(exc.stderr)
            if stderr:
                stderr = f"{stderr}\n"
            stderr = f"{stderr}docker timed out after {spec.timeout_seconds}s"
            completed = subprocess.CompletedProcess(
                command, returncode=124, stdout=stdout, stderr=stderr
            )
        except OSError as exc:
            completed = subprocess.CompletedProcess(
                command, returncode=127, stdout="", stderr=str(exc)
            )

        completed.command = command  # type: ignore[attr-defined]
        return completed


def _decode_output(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)