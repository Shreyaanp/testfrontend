"""Shared controller state definitions for mdai kiosk."""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Optional


class SessionPhase(str, enum.Enum):
    IDLE = "idle"
    PAIRING_REQUEST = "pairing_request"
    QR_DISPLAY = "qr_display"
    WAITING_ACTIVATION = "waiting_activation"
    HUMAN_DETECT = "human_detect"
    STABILIZING = "stabilizing"
    UPLOADING = "uploading"
    WAITING_ACK = "waiting_ack"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ControllerEvent:
    """Event payload distributed to UI clients over the local WebSocket."""

    type: str
    data: Dict[str, Any]
    phase: SessionPhase
    error: Optional[str] = None


__all__ = ["SessionPhase", "ControllerEvent"]
