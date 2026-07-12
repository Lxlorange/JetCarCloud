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
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
        )
