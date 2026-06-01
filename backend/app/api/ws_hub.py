from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

from fastapi import WebSocket


class WsHub:
    """
    Simple in-process websocket hub.
    For local demo it's enough; later can swap to Redis pubsub.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._classroom_sockets: dict[int, set[WebSocket]] = defaultdict(set)

    async def connect(self, classroom_id: int, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._classroom_sockets[classroom_id].add(ws)

    async def disconnect(self, classroom_id: int, ws: WebSocket) -> None:
        async with self._lock:
            self._classroom_sockets[classroom_id].discard(ws)

    async def broadcast(self, classroom_id: int, type_: str, payload: dict[str, Any]) -> None:
        event = {"type": type_, "payload": payload, "ts": datetime.utcnow().isoformat()}
        async with self._lock:
            sockets = list(self._classroom_sockets.get(classroom_id, set()))
        if not sockets:
            return
        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._classroom_sockets[classroom_id].discard(ws)


ws_hub = WsHub()

