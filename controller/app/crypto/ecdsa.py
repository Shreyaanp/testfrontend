"""ECDSA signing helpers for nonce authentication."""
from __future__ import annotations

import logging
from functools import lru_cache

from eth_account import Account
from eth_account.messages import encode_defunct

from ..config import Settings

logger = logging.getLogger(__name__)


@lru_cache()
def _load_account(private_key: str):
    try:
        return Account.from_key(private_key)
    except Exception as exc:  # pragma: no cover - fails only on misconfiguration
        raise ValueError("Invalid private key supplied for ECDSA signing") from exc


def sign_nonce(settings: Settings, nonce: str) -> str:
    """Produce an Ethereum-compatible ECDSA signature over the backend nonce."""

    account = _load_account(settings.private_key)
    message = encode_defunct(text=nonce)
    signed = Account.sign_message(message, private_key=account.key)
    signature_hex = signed.signature.hex()

    derived_address = signed.address.lower()
    expected_address = settings.evm_address.lower()
    if derived_address != expected_address:
        logger.warning(
            "Derived signer address %s does not match configured address %s",
            derived_address,
            expected_address,
        )

    return signature_hex


__all__ = ["sign_nonce"]
