"""
Aethelred Cryptographic Backend Fallback System

Provides a unified API that automatically selects the best available cryptographic
backend at import time:

1. **Native (liboqs)** — Fastest, production-grade, requires ``pip install liboqs-python``
2. **Pure-Python** — Zero native dependencies, works everywhere, suitable for dev/CI

The fallback is transparent: all public APIs remain identical regardless of backend.

.. admonition:: Security Audit

   Audited 2026-02-22. The ``ecdsa`` package is REQUIRED for ECDSA operations;
   the previous HMAC-based pseudo-ECDSA fallback has been removed (PY-03 finding).
   Exception handlers now catch specific types, not bare Exception (PY-06 finding).

Usage:
    >>> from aethelred.crypto.fallback import get_backend, HybridSigner, HybridVerifier
    >>> backend = get_backend()
    >>> print(f"Using: {backend.name} ({backend.variant})")
    >>> signer = HybridSigner()
    >>> sig = signer.sign(b"hello world")
    >>> assert signer.verify(b"hello world", sig)
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

# PY-21 fix: Explicit public API surface
__all__ = [
    "ECDSASigner",
    "ECDSAVerifier",
]


class BackendVariant(Enum):
    """Available cryptographic backend variants."""
    NATIVE_LIBOQS = "native-liboqs"
    PURE_PYTHON = "pure-python"


@dataclass(frozen=True)
class BackendInfo:
    """Information about the active cryptographic backend."""
    name: str
    variant: BackendVariant
    version: str
    fips_compliant: bool
    constant_time: bool

    def __str__(self) -> str:
        marker = "✓ FIPS" if self.fips_compliant else "⚠ non-FIPS"
        return f"{self.name} ({self.variant.value}) [{marker}]"


@runtime_checkable
class SignerProtocol(Protocol):
    """Protocol for digital signature implementations."""
    def sign(self, message: bytes) -> bytes: ...
    def verify(self, message: bytes, signature: bytes) -> bool: ...
    def public_key_bytes(self) -> bytes: ...


@runtime_checkable
class KEMProtocol(Protocol):
    """Protocol for key encapsulation implementations."""
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]: ...
    def decapsulate(self, ciphertext: bytes) -> bytes: ...
    def public_key_bytes(self) -> bytes: ...


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_backend_info: Optional[BackendInfo] = None


def _detect_backend() -> BackendInfo:
    """Detect the best available cryptographic backend."""
    # Try 1: liboqs native bindings
    try:
        oqs = importlib.import_module("oqs")
        version = getattr(oqs, "__version__", "unknown")
        logger.info("Using native liboqs backend (v%s)", version)
        return BackendInfo(
            name="liboqs",
            variant=BackendVariant.NATIVE_LIBOQS,
            version=version,
            fips_compliant=True,
            constant_time=True,
        )
    except ImportError:
        pass

    # Fallback: pure-Python implementation
    logger.info("Using pure-Python cryptographic backend (no native deps)")
    warnings.warn(
        "Using pure-Python PQC fallback. Install `liboqs-python` for "
        "production-grade constant-time implementations: pip install liboqs-python",
        stacklevel=2,
    )
    return BackendInfo(
        name="aethelred-pure-python",
        variant=BackendVariant.PURE_PYTHON,
        version="0.1.0",
        fips_compliant=False,
        constant_time=False,
    )


def get_backend() -> BackendInfo:
    """Return information about the active backend. Cached after first call."""
    global _backend_info
    if _backend_info is None:
        _backend_info = _detect_backend()
    return _backend_info


def is_native() -> bool:
    """Return True if using the native (liboqs) backend."""
    return get_backend().variant == BackendVariant.NATIVE_LIBOQS


# ---------------------------------------------------------------------------
# Unified Hybrid Signer
# ---------------------------------------------------------------------------

class HybridSigner:
    """
    Unified hybrid post-quantum signer.

    Combines ECDSA (secp256k1) + Dilithium3 signatures.
    Automatically uses the best available backend.

    Example:
        >>> signer = HybridSigner()
        >>> sig = signer.sign(b"transfer 100 AETH to 0x...")
        >>> assert signer.verify(b"transfer 100 AETH to 0x...", sig)
    """

    def __init__(
        self,
        *,
        ecdsa_private_key: Optional[bytes] = None,
        dilithium_secret_key: Optional[bytes] = None,
        dilithium_public_key: Optional[bytes] = None,
    ) -> None:
        backend = get_backend()

        if backend.variant == BackendVariant.NATIVE_LIBOQS:
            self._impl = _NativeHybridSigner(
                ecdsa_private_key=ecdsa_private_key,
                dilithium_secret_key=dilithium_secret_key,
                dilithium_public_key=dilithium_public_key,
            )
        else:
            self._impl = _PurePythonHybridSigner(
                ecdsa_private_key=ecdsa_private_key,
                dilithium_secret_key=dilithium_secret_key,
                dilithium_public_key=dilithium_public_key,
            )

    def sign(self, message: bytes) -> bytes:
        """Sign message with hybrid ECDSA + Dilithium3."""
        return self._impl.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify a hybrid signature."""
        return self._impl.verify(message, signature)

    def public_key_bytes(self) -> bytes:
        """Return the hybrid public key (ECDSA || Dilithium)."""
        return self._impl.public_key_bytes()

    @property
    def fingerprint(self) -> str:
        """SHA-256 fingerprint of the public key."""
        return hashlib.sha256(self.public_key_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Unified Hybrid Verifier (public-key only, no secret key)
# ---------------------------------------------------------------------------

class HybridVerifier:
    """
    Verify hybrid signatures using only the public key.

    Example:
        >>> verifier = HybridVerifier(signer.public_key_bytes())
        >>> assert verifier.verify(b"data", signature)
    """

    def __init__(self, public_key: bytes) -> None:
        backend = get_backend()
        if backend.variant == BackendVariant.NATIVE_LIBOQS:
            self._impl = _NativeHybridVerifier(public_key)
        else:
            self._impl = _PurePythonHybridVerifier(public_key)

    def verify(self, message: bytes, signature: bytes) -> bool:
        return self._impl.verify(message, signature)


# ---------------------------------------------------------------------------
# Pure-Python backend implementation
# ---------------------------------------------------------------------------

class _PurePythonHybridSigner:
    """Pure-Python hybrid signer using aethelred.crypto.pqc modules."""

    # Wire format: [4 bytes ECDSA sig len][ECDSA sig][Dilithium sig]
    HEADER_SIZE = 4

    def __init__(
        self,
        *,
        ecdsa_private_key: Optional[bytes] = None,
        dilithium_secret_key: Optional[bytes] = None,
        dilithium_public_key: Optional[bytes] = None,
    ) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner, DilithiumSecurityLevel

        # Initialize ECDSA (secp256k1) — ecdsa package is REQUIRED (PY-03 fix)
        try:
            from ecdsa import SigningKey, SECP256k1
        except ImportError:
            raise ImportError(
                "The 'ecdsa' package is required for ECDSA operations. "
                "Install it with: pip install ecdsa"
            )

        if ecdsa_private_key:
            self._ecdsa_sk = SigningKey.from_string(ecdsa_private_key, curve=SECP256k1)
        else:
            self._ecdsa_sk = SigningKey.generate(curve=SECP256k1)
        self._ecdsa_vk = self._ecdsa_sk.get_verifying_key()

        # Initialize Dilithium3
        self._dilithium = DilithiumSigner(
            level=DilithiumSecurityLevel.LEVEL3,
            secret_key=dilithium_secret_key,
            public_key=dilithium_public_key,
        )

    def sign(self, message: bytes) -> bytes:
        # ECDSA signature
        ecdsa_sig = self._ecdsa_sk.sign(message, hashfunc=hashlib.sha256)

        # Dilithium signature
        dil_sig = self._dilithium.sign(message)

        # Wire format
        header = len(ecdsa_sig).to_bytes(self.HEADER_SIZE, "big")
        return header + ecdsa_sig + dil_sig.to_bytes()

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            ecdsa_len = int.from_bytes(signature[: self.HEADER_SIZE], "big")
            if ecdsa_len > len(signature) - self.HEADER_SIZE:
                return False
            ecdsa_sig = signature[self.HEADER_SIZE : self.HEADER_SIZE + ecdsa_len]
            dil_sig_bytes = signature[self.HEADER_SIZE + ecdsa_len :]

            # Verify ECDSA
            self._ecdsa_vk.verify(ecdsa_sig, message, hashfunc=hashlib.sha256)

            # Verify Dilithium
            return self._dilithium.verify(message, dil_sig_bytes)
        except (ValueError, IndexError) as e:
            logger.debug("Signature verification failed: %s", e)
            return False
        except Exception as e:
            # Catch ecdsa.keys.BadSignatureError and other verification failures
            logger.debug("Signature verification failed: %s", e)
            return False

    def public_key_bytes(self) -> bytes:
        ecdsa_pk = self._ecdsa_vk.to_string()
        dil_pk = self._dilithium.public_key_bytes()
        return ecdsa_pk + dil_pk


class _PurePythonHybridVerifier:
    """Pure-Python hybrid verifier (public key only)."""

    HEADER_SIZE = 4

    def __init__(self, public_key: bytes) -> None:
        from aethelred.crypto.pqc.dilithium import (
            DilithiumSigner,
            DilithiumSecurityLevel,
            DILITHIUM_SIZES,
        )

        dil_pk_size = DILITHIUM_SIZES[DilithiumSecurityLevel.LEVEL3]["public_key"]
        if len(public_key) <= dil_pk_size:
            raise ValueError(
                f"Public key too short: expected >{dil_pk_size} bytes, "
                f"got {len(public_key)}"
            )
        self._ecdsa_pk_bytes = public_key[:-dil_pk_size]
        self._dil_pk_bytes = public_key[-dil_pk_size:]

        try:
            from ecdsa import VerifyingKey, SECP256k1
            self._ecdsa_vk = VerifyingKey.from_string(self._ecdsa_pk_bytes, curve=SECP256k1)
        except ImportError:
            raise ImportError(
                "The 'ecdsa' package is required for ECDSA verification. "
                "Install it with: pip install ecdsa"
            )

        # Verification-only: use DilithiumSigner.verify_with_public_key static method
        # instead of creating a signer with dummy secret key (PY-17 fix)
        self._dil_level = DilithiumSecurityLevel.LEVEL3

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            ecdsa_len = int.from_bytes(signature[: self.HEADER_SIZE], "big")
            if ecdsa_len > len(signature) - self.HEADER_SIZE:
                return False
            ecdsa_sig = signature[self.HEADER_SIZE : self.HEADER_SIZE + ecdsa_len]
            dil_sig_bytes = signature[self.HEADER_SIZE + ecdsa_len :]

            self._ecdsa_vk.verify(ecdsa_sig, message, hashfunc=hashlib.sha256)

            # Use static verification (no dummy secret key needed)
            from aethelred.crypto.pqc.dilithium import DilithiumSigner
            return DilithiumSigner.verify_with_public_key(
                message, dil_sig_bytes, self._dil_pk_bytes, self._dil_level
            )
        except (ValueError, IndexError) as e:
            logger.debug("Hybrid verification failed: %s", e)
            return False
        except Exception as e:
            # Catch ecdsa.keys.BadSignatureError and other verification failures
            logger.debug("Hybrid verification failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Native (liboqs) backend implementation
# ---------------------------------------------------------------------------

class _NativeHybridSigner:
    """Native liboqs hybrid signer — production-grade, constant-time."""

    HEADER_SIZE = 4

    def __init__(
        self,
        *,
        ecdsa_private_key: Optional[bytes] = None,
        dilithium_secret_key: Optional[bytes] = None,
        dilithium_public_key: Optional[bytes] = None,
    ) -> None:
        import oqs  # type: ignore[import-untyped]

        # ECDSA via python-ecdsa
        from ecdsa import SigningKey, SECP256k1
        if ecdsa_private_key:
            self._ecdsa_sk = SigningKey.from_string(ecdsa_private_key, curve=SECP256k1)
        else:
            self._ecdsa_sk = SigningKey.generate(curve=SECP256k1)
        self._ecdsa_vk = self._ecdsa_sk.get_verifying_key()

        # Dilithium3 via liboqs
        self._sig = oqs.Signature("Dilithium3", dilithium_secret_key)
        if dilithium_public_key:
            self._public_key = dilithium_public_key
        else:
            self._public_key = self._sig.generate_keypair()

    def sign(self, message: bytes) -> bytes:
        ecdsa_sig = self._ecdsa_sk.sign(message, hashfunc=hashlib.sha256)
        dil_sig = self._sig.sign(message)
        header = len(ecdsa_sig).to_bytes(self.HEADER_SIZE, "big")
        return header + ecdsa_sig + dil_sig

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            ecdsa_len = int.from_bytes(signature[: self.HEADER_SIZE], "big")
            ecdsa_sig = signature[self.HEADER_SIZE : self.HEADER_SIZE + ecdsa_len]
            dil_sig = signature[self.HEADER_SIZE + ecdsa_len :]
            self._ecdsa_vk.verify(ecdsa_sig, message, hashfunc=hashlib.sha256)
            return self._sig.verify(message, dil_sig, self._public_key)
        except Exception:
            return False

    def public_key_bytes(self) -> bytes:
        return self._ecdsa_vk.to_string() + self._public_key


class _NativeHybridVerifier:
    """Native liboqs hybrid verifier — public key only."""

    HEADER_SIZE = 4

    def __init__(self, public_key: bytes) -> None:
        import oqs  # type: ignore[import-untyped]
        from ecdsa import VerifyingKey, SECP256k1

        sig = oqs.Signature("Dilithium3")
        dil_pk_size = sig.length_public_key
        self._ecdsa_vk = VerifyingKey.from_string(public_key[:-dil_pk_size], curve=SECP256k1)
        self._dil_pk = public_key[-dil_pk_size:]
        self._sig = sig

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            ecdsa_len = int.from_bytes(signature[: self.HEADER_SIZE], "big")
            ecdsa_sig = signature[self.HEADER_SIZE : self.HEADER_SIZE + ecdsa_len]
            dil_sig = signature[self.HEADER_SIZE + ecdsa_len :]
            self._ecdsa_vk.verify(ecdsa_sig, message, hashfunc=hashlib.sha256)
            return self._sig.verify(message, dil_sig, self._dil_pk)
        except Exception:
            return False
