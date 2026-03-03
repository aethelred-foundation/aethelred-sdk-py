"""Utility functions for Aethelred SDK.

Provides common helpers used across modules:
- Hashing (SHA-256, content-addressable)
- Byte/hex conversions
- Retry logic with exponential backoff
- Size formatting
"""

from __future__ import annotations

import hashlib
import time
import logging
import base64
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

__all__ = [
    "sha256",
    "sha256_hex",
    "sha256_bytes",
    "keccak256",
    "bytes_to_hex",
    "hex_to_bytes",
    "to_uaethel",
    "from_uaethel",
    "format_aethel",
    "is_valid_address",
    "encode_base64",
    "decode_base64",
    "format_size",
    "retry",
]

T = TypeVar("T")


def sha256_hex(data: Union[bytes, str]) -> str:
    """Compute SHA-256 and return hex digest.

    Args:
        data: Input bytes or UTF-8 string

    Returns:
        64-character hex string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: Union[bytes, str]) -> bytes:
    """Compute SHA-256 and return raw digest (32 bytes).

    Args:
        data: Input bytes or UTF-8 string

    Returns:
        32-byte digest
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).digest()


def sha256(data: Union[bytes, str]) -> bytes:
    """Backward-compatible alias returning raw SHA-256 digest bytes."""
    return sha256_bytes(data)


def keccak256(data: Union[bytes, str]) -> bytes:
    """Best-effort keccak256-compatible helper using SHA3-256 from stdlib."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha3_256(data).digest()


def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to lowercase hex string."""
    return data.hex()


def hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes.

    Raises:
        ValueError: If hex_str is not valid hex
    """
    return bytes.fromhex(hex_str)


def to_uaethel(amount: Union[int, float]) -> int:
    """Convert AETHEL to uaethel (1e6 base units)."""
    return int(round(float(amount) * 1_000_000))


def from_uaethel(amount_uaethel: int) -> float:
    """Convert uaethel to AETHEL."""
    return float(amount_uaethel) / 1_000_000.0


def format_aethel(amount_uaethel: int, decimals: int = 6) -> str:
    """Format base units as human-readable AETHEL string."""
    return f"{from_uaethel(amount_uaethel):.{decimals}f} AETHEL"


def is_valid_address(address: str) -> bool:
    """Lightweight Aethelred bech32-style address shape validation."""
    return isinstance(address, str) and address.startswith("aethel1") and len(address) >= 16


def encode_base64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode("ascii")


def decode_base64(value: str) -> bytes:
    """Decode base64 string to bytes."""
    return base64.b64decode(value.encode("ascii"))


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string.

    Examples:
        >>> format_size(0)
        '0 B'
        >>> format_size(1024)
        '1.0 KiB'
        >>> format_size(1048576)
        '1.0 MiB'
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024.0 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_index]}"


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        backoff_base: Base delay in seconds
        backoff_max: Maximum delay in seconds
        exceptions: Exception types to catch

    Example::

        @retry(max_attempts=3, backoff_base=0.5)
        def unreliable_call():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = min(backoff_base * (2 ** attempt), backoff_max)
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1, max_attempts, func.__name__, delay, e,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
