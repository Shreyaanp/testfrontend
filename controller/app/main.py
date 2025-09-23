"""FastAPI entry-point for the mdai controller."""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings, get_settings
from .logging_config import configure_logging
from .session_manager import SessionManager

settings: Settings = get_settings()
configure_logging(settings.log_level)
app = FastAPI(title="mdai-controller", version="0.1.0")
manager = SessionManager(settings=settings)


@app.on_event("startup")
async def on_startup() -> None:
    await manager.start()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await manager.stop()


@app.get("/healthz")
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok", "phase": manager.phase.value})


@app.post("/debug/trigger")
async def debug_trigger() -> JSONResponse:
    await manager.trigger_debug_session()
    return JSONResponse({"status": "scheduled"})


@app.get("/preview")
async def preview_stream() -> StreamingResponse:
    boundary = "frame"

    async def frame_iterator() -> AsyncIterator[bytes]:
        async for frame in manager.preview_frames():
            header = (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(frame)}\r\n\r\n"
            ).encode("ascii")
            yield header + frame + b"\r\n"

    media_type = f"multipart/x-mixed-replace; boundary={boundary}"
    return StreamingResponse(frame_iterator(), media_type=media_type)


@app.websocket("/ws/ui")
async def ui_socket(ws: WebSocket) -> None:
    await ws.accept()
    queue = manager.register_ui()
    try:
        while True:
            event = await queue.get()
            payload = {
                "type": event.type,
                "phase": event.phase.value,
                "data": event.data,
            }
            if event.error:
                payload["error"] = event.error
            await ws.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        manager.unregister_ui(queue)
        await ws.close()
