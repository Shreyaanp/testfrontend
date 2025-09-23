"""HTTP client for mdai pairing endpoints."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from ..config import Settings

logger = logging.getLogger(__name__)


class PairingHttpClient:
    """Thin wrapper around backend `/pair/*` endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = httpx.AsyncClient(base_url=settings.backend_api_url, timeout=15.0)

    async def prepare(self) -> Optional[str]:
        payload = {"device_id": self.settings.device_id}
        resp = await self._client.post("/pair/prepare", json=payload)
        resp.raise_for_status()
        data = resp.json()
        nonce = data.get("nonce")
        if not nonce:
            logger.error("Pair prepare response missing nonce: %s", data)
            return None
        return nonce

    async def request(self, signature: str) -> Dict[str, Any]:
        payload = {
            "device_id": self.settings.device_id,
            "signature": signature,
        }
        resp = await self._client.post("/pair/request", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()

