from __future__ import annotations

import asyncio
import json
import socket


async def broadcast_discovery_beacon(
    *,
    service: str,
    host: str,
    port: int,
    http_port: int,
    interval_seconds: float,
) -> None:
    payload = json.dumps(
        {
            "service": service,
            "host": host,
            "port": http_port,
            "scheme": "http",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            sock.sendto(payload, ("255.255.255.255", int(port)))
            await asyncio.sleep(interval_seconds)
