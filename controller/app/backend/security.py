"""Security helpers for backend communication."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any, Dict


def _decode_session_key(session_key: str) -> bytes:
    key = session_key.strip()
    # Try base64 decode first (common for random session keys)
    for decoder in (_decode_base64, _decode_hex):
        material = decoder(key)
        if material is not None:
            return material
    return key.encode("utf-8")


def _decode_base64(value: str) -> bytes | None:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def _decode_hex(value: str) -> bytes | None:
    try:
        return bytes.fromhex(value.removeprefix("0x"))
    except ValueError:
        return None


def canonical_payload(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def compute_hmac(session_key: str, payload: Dict[str, Any]) -> str:
    key_bytes = _decode_session_key(session_key)
    message = canonical_payload(payload)
    return hmac.new(key_bytes, message, hashlib.sha256).hexdigest()


__all__ = ["compute_hmac", "canonical_payload"]
