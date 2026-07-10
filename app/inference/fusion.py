from __future__ import annotations

from typing import Any


def estimate_distance_m(bbox: list[float], sensors: dict[str, Any]) -> float | None:
    lidar = sensors.get("lidar") if isinstance(sensors, dict) else None
    if not isinstance(lidar, dict):
        return None

    ranges = lidar.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        return None

    valid = [float(v) for v in ranges if isinstance(v, (int, float)) and v > 0]
    if not valid:
        return None

    # Placeholder fusion: use the nearest lidar reading until camera-lidar
    # calibration is available. The API already exposes distance_m so the app
    # contract will not need to change when this becomes angle-aware.
    return min(valid)

