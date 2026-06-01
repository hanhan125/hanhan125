from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.camera import router as camera_router
from app.api.demo import router as demo_router
from app.api.routes import router as api_router
from app.api.ws_hub import ws_hub
from app.core.config import settings
from app.storage.db import init_db


def _parse_cors() -> list[str]:
    parts = [p.strip() for p in settings.cors_origins.split(",")]
    return [p for p in parts if p]


def _cors_regex() -> str | None:
    r = (settings.cors_origin_regex or "").strip()
    return r or None


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors(),
    allow_origin_regex=_cors_regex(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(demo_router)
app.include_router(camera_router)


@app.on_event("startup")
def _on_startup() -> None:
    init_db()


@app.websocket("/ws/classrooms/{classroom_id}")
async def classroom_ws(ws: WebSocket, classroom_id: int) -> None:
    await ws_hub.connect(classroom_id, ws)
    try:
        while True:
            # Keep connection alive; allow client pings if needed.
            _ = await ws.receive_text()
    except WebSocketDisconnect:
        await ws_hub.disconnect(classroom_id, ws)
    except Exception:
        await ws_hub.disconnect(classroom_id, ws)

