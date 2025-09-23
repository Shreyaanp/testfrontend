"""Backend WebSocket client used during active sessions."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import websockets

from ..config import Settings

logger = logging.getLogger(__name__)

IncomingHandler = Callable[[dict[str, Any]], Awaitable[None]]


class BackendWebSocketClient:
    """Maintains connection to backend `/ws/device/{device_id}`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._uri = f"{settings.backend_ws_url}/device/{settings.device_id}"
        self._conn: Optional[websockets.client.WebSocketClientProtocol] = None
        self._listener_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    async def connect(self, handler: IncomingHandler) -> None:
        await self.disconnect()
        logger.info("Connecting to backend websocket %s", self._uri)
        self._stop_event.clear()
        self._conn = await websockets.connect(self._uri, ping_interval=30, ping_timeout=10)
        self._listener_task = asyncio.create_task(self._listen(handler), name="backend-ws-listener")

    async def disconnect(self) -> None:
        self._stop_event.set()
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        self._listener_task = None
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _listen(self, handler: IncomingHandler) -> None:
        assert self._conn is not None
        try:
            async for message in self._conn:
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from backend: %s", message)
                    continue
                await handler(payload)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            raise
        except websockets.ConnectionClosedOK:
            logger.info("Backend websocket closed cleanly")
        except websockets.ConnectionClosedError as exc:
            logger.warning("Backend websocket closed: %s", exc)
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Backend websocket listener crashed")
        finally:
            self._stop_event.set()
            self._listener_task = None
            if self._conn:
                await self._conn.close()
                self._conn = None

    async def send(self, message: dict[str, Any]) -> None:
        if not self._conn:
            raise RuntimeError("Backend websocket not connected")
        await self._conn.send(json.dumps(message))
