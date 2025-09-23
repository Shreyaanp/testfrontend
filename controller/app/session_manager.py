"""Session orchestration for mdai kiosk."""
from __future__ import annotations

import asyncio
from asyncio import QueueEmpty
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from .backend.http_client import PairingHttpClient
from .backend.ws_client import BackendWebSocketClient
from .config import Settings, get_settings
from .crypto.ecdsa import sign_nonce
from .backend.security import compute_hmac
from .sensors.realsense import RealSenseService
from .sensors.tof import ToFSensor, mock_distance_provider
from .state import ControllerEvent, SessionPhase

logger = logging.getLogger(__name__)


@dataclass
class SessionContext:
    pairing_token: Optional[str] = None
    expires_in: Optional[int] = None
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    latest_distance_mm: Optional[int] = None
    best_frame_b64: Optional[str] = None
    next_seq: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Coordinates sensors, backend comms, and UI state updates."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        tof_distance_provider=mock_distance_provider,
    ) -> None:
        self.settings = settings or get_settings()
        self._lock = asyncio.Lock()
        self._phase: SessionPhase = SessionPhase.IDLE
        self._ui_subscribers: List[asyncio.Queue[ControllerEvent]] = []
        self._current_session = SessionContext()

        self._tof = ToFSensor(
            threshold_mm=self.settings.tof_threshold_mm,
            debounce_ms=self.settings.tof_debounce_ms,
            distance_provider=tof_distance_provider,
        )
        self._tof.register_callback(self._handle_tof_trigger)

        self._realsense = RealSenseService()
        self._http_client = PairingHttpClient(self.settings)
        self._ws_client = BackendWebSocketClient(self.settings)

        self._session_task: Optional[asyncio.Task[None]] = None
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._session_active_event: Optional[asyncio.Event] = None
        self._ack_event: Optional[asyncio.Event] = None
        self._last_metrics_ts: float = 0.0

    @property
    def phase(self) -> SessionPhase:
        return self._phase

    async def start(self) -> None:
        logger.info("Starting session manager")
        await self._realsense.start()
        await self._tof.start()
        self._background_tasks.append(asyncio.create_task(self._heartbeat_loop(), name="controller-heartbeat"))

    async def stop(self) -> None:
        logger.info("Stopping session manager")
        await self._tof.stop()
        await self._realsense.stop()
        for task in self._background_tasks:
            task.cancel()
        for task in self._background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._background_tasks.clear()
        await self._ws_client.disconnect()
        await self._http_client.aclose()

    def register_ui(self) -> asyncio.Queue[ControllerEvent]:
        queue: asyncio.Queue[ControllerEvent] = asyncio.Queue(maxsize=4)
        self._ui_subscribers.append(queue)
        return queue

    def unregister_ui(self, queue: asyncio.Queue[ControllerEvent]) -> None:
        if queue in self._ui_subscribers:
            self._ui_subscribers.remove(queue)

    async def trigger_debug_session(self) -> None:
        """Trigger a session manually (used by debug endpoint)."""

        self._schedule_session()

    async def preview_frames(self) -> AsyncIterator[bytes]:
        async for frame in self._realsense.preview_stream():
            yield frame

    async def _broadcast(self, event: ControllerEvent) -> None:
        logger.debug("Broadcasting event: %s", event)
        for queue in list(self._ui_subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def _set_phase(self, phase: SessionPhase, *, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        self._phase = phase
        await self._broadcast(ControllerEvent(type="state", data=data or {}, phase=phase, error=error))

    async def _handle_tof_trigger(self, triggered: bool, distance: int) -> None:
        logger.info("ToF trigger=%s distance=%s phase=%s", triggered, distance, self.phase)
        self._current_session.latest_distance_mm = distance
        if triggered and self.phase == SessionPhase.IDLE:
            self._schedule_session()
        elif not triggered and self.phase not in {SessionPhase.IDLE, SessionPhase.COMPLETE}:
            logger.info("ToF reset detected mid-session; cancelling")
            if self._session_task:
                self._session_task.cancel()

    def _schedule_session(self) -> None:
        if self._session_task and not self._session_task.done():
            logger.info("Existing session task running; ignoring new trigger")
            return
        self._session_task = asyncio.create_task(self._run_session(), name="controller-session")

    async def _run_session(self) -> None:
        try:
            await self._set_phase(SessionPhase.PAIRING_REQUEST)
            nonce = await self._http_client.prepare()
            if not nonce:
                await self._set_phase(SessionPhase.ERROR, error="missing_nonce")
                return

            signature = sign_nonce(self.settings, nonce)
            pairing_data = await self._http_client.request(signature)
            token = pairing_data.get("pairing_token")
            expires_in = pairing_data.get("expires_in")
            if not token:
                await self._set_phase(SessionPhase.ERROR, error="missing_pairing_token")
                return

            self._current_session = SessionContext(pairing_token=token, expires_in=expires_in)
            self._last_metrics_ts = 0.0
            await self._set_phase(SessionPhase.QR_DISPLAY, data={"pairing_token": token, "expires_in": expires_in})

            self._session_active_event = asyncio.Event()
            self._ack_event = asyncio.Event()
            await self._ws_client.connect(self._handle_backend_message)

            await self._await_session_activation()
            await self._set_phase(SessionPhase.HUMAN_DETECT)
            await self._collect_best_frame()
            await self._upload_frame()
            await self._wait_for_ack()
            await self._set_phase(SessionPhase.COMPLETE)
        except asyncio.CancelledError:
            logger.info("Session cancelled")
            await self._set_phase(SessionPhase.IDLE)
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Session failed: %s", exc)
            await self._set_phase(SessionPhase.ERROR, error=str(exc))
        finally:
            await self._ws_client.disconnect()
            await asyncio.sleep(1.0)
            await self._set_phase(SessionPhase.IDLE)
            if self._session_task:
                self._session_task = None
            self._session_active_event = None
            self._ack_event = None

    async def _await_session_activation(self) -> None:
        await self._set_phase(SessionPhase.WAITING_ACTIVATION)
        if not self._session_active_event:
            raise RuntimeError("session_activation_event_not_initialized")
        await asyncio.wait_for(self._session_active_event.wait(), timeout=60.0)

    async def _collect_best_frame(self) -> None:
        await self._set_phase(SessionPhase.STABILIZING)
        results = await self._realsense.gather_results(self.settings.stability_seconds)
        if not results:
            raise RuntimeError("liveness_capture_failed")

        best_bytes: Optional[bytes] = None
        best_score = -1.0

        for result in results:
            if not (result.instant_alive or result.stable_alive):
                continue
            focus_score = self._compute_focus(result.color_image)
            normalized_focus = min(focus_score / 800.0, 1.0)
            stability = result.stability_score
            composite = (stability * 0.7) + (normalized_focus * 0.3)
            if result.stable_alive:
                composite += 0.05
            if composite > best_score:
                encoded = self._encode_jpeg(result.color_image)
                if encoded is None:
                    continue
                best_bytes = encoded
                best_score = composite
            now = time.time()
            if now - self._last_metrics_ts >= 0.2:
                self._last_metrics_ts = now
                await self._broadcast(
                    ControllerEvent(
                        type="metrics",
                        phase=self._phase,
                        data={
                            "stability": stability,
                            "focus": focus_score,
                            "composite": composite,
                        },
                    )
                )

        if not best_bytes:
            raise RuntimeError("no_viable_frame")

        self._current_session.best_frame_b64 = base64.b64encode(best_bytes).decode()

    async def _upload_frame(self) -> None:
        await self._set_phase(SessionPhase.UPLOADING)
        if not self._current_session.best_frame_b64:
            raise RuntimeError("no_frame_to_upload")
        if not self._current_session.session_id or not self._current_session.session_key:
            raise RuntimeError("session_not_activated")

        payload_body = {
            "snapshot_base64": self._current_session.best_frame_b64,
        }
        seq = self._current_session.next_seq
        timestamp_ms = int(time.time() * 1000)
        envelope_core = {
            "session_id": self._current_session.session_id,
            "seq": seq,
            "timestamp": timestamp_ms,
            "payload": payload_body,
        }
        hmac_value = compute_hmac(self._current_session.session_key, envelope_core)
        envelope = {**envelope_core, "hmac": hmac_value}
        payload = {
            "type": "session_message",
            "envelope": envelope,
        }
        await self._ws_client.send(payload)
        self._current_session.next_seq += 1

    async def _wait_for_ack(self) -> None:
        await self._set_phase(SessionPhase.WAITING_ACK)
        if not self._ack_event:
            raise RuntimeError("ack_event_not_initialized")
        await asyncio.wait_for(self._ack_event.wait(), timeout=120.0)

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(30)
                await self._broadcast(ControllerEvent(type="heartbeat", data={}, phase=self.phase))
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            raise

    async def _handle_backend_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        logger.info("Backend message received: %s", message_type)
        if message_type == "session_activated":
            self._current_session.session_id = message.get("session_id")
            self._current_session.session_key = message.get("session_key")
            if self._session_active_event and not self._session_active_event.is_set():
                self._session_active_event.set()
            await self._broadcast(
                ControllerEvent(
                    type="backend",
                    phase=self._phase,
                    data={"event": "session_activated"},
                )
            )
        elif message_type == "end_session":
            if self._ack_event and not self._ack_event.is_set():
                self._ack_event.set()
            await self._broadcast(
                ControllerEvent(
                    type="backend",
                    phase=self._phase,
                    data={"event": "end_session", "reason": message.get("reason")},
                )
            )
        elif message_type == "session_message":
            await self._broadcast(
                ControllerEvent(
                    type="backend",
                    phase=self._phase,
                    data={"event": "session_message", "payload": message.get("envelope")},
                )
            )
        elif message_type == "hardware_register_result":
            await self._broadcast(
                ControllerEvent(
                    type="backend",
                    phase=self._phase,
                    data={"event": "hardware_register_result", "ok": message.get("ok")},
                )
            )
        else:
            logger.debug("Unhandled backend message: %s", message)

    @staticmethod
    def _compute_focus(image) -> float:
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:  # pragma: no cover - focus metric is best effort
            logger.exception("Failed to compute focus metric")
            return 0.0

    @staticmethod
    def _encode_jpeg(image) -> Optional[bytes]:
        try:
            import cv2

            success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                return None
            return encoded.tobytes()
        except Exception:
            logger.exception("Failed to encode JPEG frame")
            return None
