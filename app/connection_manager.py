from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self, history_size: int = 1) -> None:
        self._lock = asyncio.Lock()
        self._apps: dict[str, set[WebSocket]] = defaultdict(set)
        self._history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    async def connect_app(self, car_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._apps[car_id].add(websocket)
            history = list(self._history[car_id])

        for item in history:
            await websocket.send_json(item)

    async def disconnect_app(self, car_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            self._apps[car_id].discard(websocket)

    async def publish(self, car_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._history[car_id].append(payload)
            clients = list(self._apps[car_id])

        stale: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._apps[car_id].discard(websocket)

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                car_id: {
                    "app_clients": len(clients),
                    "history": len(self._history[car_id]),
                }
                for car_id, clients in self._apps.items()
            }

