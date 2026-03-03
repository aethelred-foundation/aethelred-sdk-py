"""
Kyber Post-Quantum Key Encapsulation Mechanism (KEM)

NIST FIPS 203 standardized lattice-based key encapsulation mechanism.
Provides NIST security levels 2, 3, and 5 (Kyber512, Kyber768, Kyber1024).

Kyber is used for:
- Secure key exchange for encrypted channels
- Wrapping symmetric keys for data encryption
- Post-quantum secure communication setup

Key Sizes (Kyber768/Level3):
    - Public Key: 1,184 bytes
    - Secret Key: 2,400 bytes
    - Ciphertext: 1,088 bytes
    - Shared Secret: 32 bytes

.. admonition:: Security Audit

   Audited 2026-02-22. Pure-Python fallback is NOT cryptographically secure;
   it exists solely for CI/dev environments. Production deployments MUST
   use liboqs (``pip install liboqs-python``).

Example:
    >>> from aethelred.crypto.pqc.kyber import KyberKEM
    >>>
    >>> # Recipient generates keypair
    >>> recipient = KyberKEM()
    >>>
    >>> # Sender encapsulates to create shared secret
    >>> ciphertext, shared_secret = KyberKEM.encapsulate_to(recipient.public_key_bytes())
    >>>
    >>> # Recipient decapsulates to derive same shared secret
    >>> derived_secret = recipient.decapsulate(ciphertext)
    >>> assert shared_secret == derived_secret
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)

# PY-21 fix: Explicit public API surface
__all__ = [
    "KyberSecurityLevel",
    "KyberKEM",
    "generate_kyber_keypair",
    "encapsulate_kyber",
    "decapsulate_kyber",
]


class KyberSecurityLevel(Enum):
    """
    Kyber security levels corresponding to NIST post-quantum security levels.

    LEVEL2: ~128-bit classical, ~64-bit quantum (AES-128 equivalent)
    LEVEL3: ~192-bit classical, ~128-bit quantum (AES-192 equivalent) [DEFAULT]
    LEVEL5: ~256-bit classical, ~128-bit quantum (AES-256 equivalent)
    """

    LEVEL2 = 2  # Kyber512
    LEVEL3 = 3  # Kyber768 (recommended)
    LEVEL5 = 5  # Kyber1024


# Key, ciphertext, and shared secret sizes for each security level
KYBER_SIZES = {
    KyberSecurityLevel.LEVEL2: {
        "public_key": 800,
        "secret_key": 1632,
        "ciphertext": 768,
        "shared_secret": 32,
    },
    KyberSecurityLevel.LEVEL3: {
        "public_key": 1184,
        "secret_key": 2400,
        "ciphertext": 1088,
        "shared_secret": 32,
    },
    KyberSecurityLevel.LEVEL5: {
        "public_key": 1568,
        "secret_key": 3168,
        "ciphertext": 1568,
        "shared_secret": 32,
    },
}


@dataclass
class KyberKeyPair:
    """Kyber key pair container."""

    public_key: bytes
    secret_key: bytes
    level: KyberSecurityLevel
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def public_key_hex(self) -> str:
        """Return public key as hex string."""
        return self.public_key.hex()

    def secret_key_hex(self) -> str:
        """Return secret key as hex string."""
        return self.secret_key.hex()

    def fingerprint(self) -> str:
        """Compute key fingerprint."""
        return hashlib.sha256(self.public_key).hexdigest()[:16]


@dataclass
class KyberCiphertext:
    """Kyber ciphertext container."""

    ciphertext: bytes
    level: KyberSecurityLevel
    recipient_fingerprint: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def hex(self) -> str:
        """Return ciphertext as hex string."""
        return self.ciphertext.hex()

    def to_bytes(self) -> bytes:
        """Return raw ciphertext bytes."""
        return self.ciphertext

    def __len__(self) -> int:
        return len(self.ciphertext)


class KyberKEM:
    """
    Kyber Key Encapsulation Mechanism implementation.

    Provides key generation, encapsulation, and decapsulation operations
    for post-quantum secure key exchange.

    Example:
        >>> # Recipient side - generate keypair
        >>> recipient = KyberKEM()
        >>> public_key = recipient.public_key_bytes()
        >>>
        >>> # Sender side - encapsulate with recipient's public key
        >>> ciphertext, shared_secret = KyberKEM.encapsulate_to(public_key)
        >>> # shared_secret can now be used as symmetric key
        >>>
        >>> # Recipient side - decapsulate to derive shared secret
        >>> derived = recipient.decapsulate(ciphertext)
        >>> assert shared_secret == derived  # Both have same secret
        >>>
        >>> # Use shared secret for AES encryption
        >>> from aethelred.crypto import encrypt_aes_gcm
        >>> ciphertext, nonce, tag = encrypt_aes_gcm(plaintext, shared_secret)
    """

    def __init__(
        self,
        level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
        secret_key: Optional[bytes] = None,
        public_key: Optional[bytes] = None,
    ):
        """
        Initialize Kyber KEM.

        Args:
            level: Security level (LEVEL2, LEVEL3, LEVEL5)
            secret_key: Existing secret key (generates new if not provided)
            public_key: Existing public key (required if secret_key provided)
        """
        self.level = level
        self._sizes = KYBER_SIZES[level]

        if secret_key is not None:
            if public_key is None:
                raise ValueError("public_key required when providing secret_key")
            self._validate_key_sizes(secret_key, public_key)
            self._secret_key = secret_key
            self._public_key = public_key
        else:
            # Generate new keypair
            keypair = generate_kyber_keypair(level)
            self._secret_key = keypair.secret_key
            self._public_key = keypair.public_key

        self._fingerprint = hashlib.sha256(self._public_key).hexdigest()[:16]

    def _validate_key_sizes(self, secret_key: bytes, public_key: bytes) -> None:
        """Validate key sizes match security level."""
        if len(secret_key) != self._sizes["secret_key"]:
            raise ValueError(
                f"Invalid secret key size: expected {self._sizes['secret_key']}, "
                f"got {len(secret_key)}"
            )
        if len(public_key) != self._sizes["public_key"]:
            raise ValueError(
                f"Invalid public key size: expected {self._sizes['public_key']}, "
                f"got {len(public_key)}"
            )

    def encapsulate(self) -> Tuple[KyberCiphertext, bytes]:
        """
        Encapsulate to own public key (for testing).

        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        return self.encapsulate_to(
            self._public_key,
            self.level,
            self._fingerprint,
        )

    @staticmethod
    def encapsulate_to(
        public_key: bytes,
        level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
        recipient_fingerprint: Optional[str] = None,
    ) -> Tuple[KyberCiphertext, bytes]:
        """
        Encapsulate a shared secret to a recipient's public key.

        This creates a ciphertext that only the private key holder can
        decapsulate to derive the same shared secret.

        Args:
            public_key: Recipient's public key
            level: Security level
            recipient_fingerprint: Optional recipient identifier

        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        ciphertext_bytes, shared_secret = encapsulate_kyber(public_key, level)

        ciphertext = KyberCiphertext(
            ciphertext=ciphertext_bytes,
            level=level,
            recipient_fingerprint=recipient_fingerprint,
        )

        return ciphertext, shared_secret

    def decapsulate(self, ciphertext: Union[KyberCiphertext, bytes]) -> bytes:
        """
        Decapsulate a ciphertext to derive the shared secret.

        Args:
            ciphertext: Ciphertext from encapsulation

        Returns:
            Shared secret (32 bytes)
        """
        ct_bytes = ciphertext.ciphertext if isinstance(ciphertext, KyberCiphertext) else ciphertext

        return decapsulate_kyber(ct_bytes, self._secret_key, self.level)

    def public_key_bytes(self) -> bytes:
        """Return public key as bytes."""
        return self._public_key

    def secret_key_bytes(self) -> bytes:
        """Return secret key as bytes."""
        return self._secret_key

    def keypair(self) -> KyberKeyPair:
        """Return full keypair."""
        return KyberKeyPair(
            public_key=self._public_key,
            secret_key=self._secret_key,
            level=self.level,
        )

    @property
    def fingerprint(self) -> str:
        """Return key fingerprint."""
        return self._fingerprint

    @classmethod
    def from_keypair(cls, keypair: KyberKeyPair) -> "KyberKEM":
        """Create KEM from existing keypair."""
        return cls(
            level=keypair.level,
            secret_key=keypair.secret_key,
            public_key=keypair.public_key,
        )

    @classmethod
    def from_hex(
        cls,
        secret_key_hex: str,
        public_key_hex: str,
        level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
    ) -> "KyberKEM":
        """Create KEM from hex-encoded keys."""
        return cls(
            level=level,
            secret_key=bytes.fromhex(secret_key_hex),
            public_key=bytes.fromhex(public_key_hex),
        )


# ============ liboqs Backend Detection ============

_LIBOQS_AVAILABLE = False
try:
    import oqs  # type: ignore[import-untyped]

    _LIBOQS_AVAILABLE = True
except ImportError:
    pass

_KYBER_OQS_NAMES = {
    KyberSecurityLevel.LEVEL2: "Kyber512",
    KyberSecurityLevel.LEVEL3: "Kyber768",
    KyberSecurityLevel.LEVEL5: "Kyber1024",
}


def generate_kyber_keypair(
    level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
) -> KyberKeyPair:
    """
    Generate a new Kyber keypair.

    Uses liboqs when available; falls back to SHAKE-256 simulation
    with a runtime warning.

    Args:
        level: Security level

    Returns:
        New keypair
    """
    if _LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation(_KYBER_OQS_NAMES[level])
        public_key = kem.generate_keypair()
        secret_key = kem.export_secret_key()
        return KyberKeyPair(
            public_key=bytes(public_key),
            secret_key=bytes(secret_key),
            level=level,
        )

    # Fallback: SHAKE-256 simulation (NOT cryptographically secure)
    if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
        raise RuntimeError(
            "CRITICAL SECURITY ERROR: liboqs not found in PRODUCTION_MODE. "
            "Cannot use insecure fallback. Install liboqs-python or check LD_LIBRARY_PATH."
        )

    warnings.warn(
        "liboqs not installed — using SHAKE-256 Kyber simulation. "
        "Install liboqs-python for real post-quantum security: pip install liboqs-python",
        stacklevel=2,
    )
    sizes = KYBER_SIZES[level]
    seed = secrets.token_bytes(64)

    # Derive keypair deterministically from seed so encaps/decaps are consistent (PY-02 fix)
    keygen_shake = hashlib.shake_256()
    keygen_shake.update(seed)
    keygen_shake.update(b"kyber_keygen_v2")
    keygen_shake.update(level.value.to_bytes(1, "big"))
    pk_size = sizes["public_key"]
    # Generate private material + public key from the same seed.
    # We store the public key in the LAST pk_size bytes of the secret key
    # so decapsulate can extract it (similar to how real Kyber sk includes pk).
    private_material_size = sizes["secret_key"] - pk_size
    key_material = keygen_shake.digest(private_material_size + pk_size)
    public_key = key_material[private_material_size:]
    # secret_key = private_material || public_key
    secret_key = key_material
    return KyberKeyPair(public_key=public_key, secret_key=secret_key, level=level)


def encapsulate_kyber(
    public_key: bytes,
    level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
) -> Tuple[bytes, bytes]:
    """
    Encapsulate to create ciphertext and shared secret.

    Uses liboqs when available; falls back to SHAKE-256 simulation.

    Args:
        public_key: Recipient's public key
        level: Security level

    Returns:
        Tuple of (ciphertext, shared_secret)
    """
    sizes = KYBER_SIZES[level]

    if len(public_key) != sizes["public_key"]:
        raise ValueError(f"Invalid public key size for Kyber{level.value}")

    if _LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation(_KYBER_OQS_NAMES[level])
        ciphertext, shared_secret = kem.encap_secret(public_key)
        return bytes(ciphertext), bytes(shared_secret)

    # Fallback: seed-based simulation that matches decapsulation (PY-02 fix)
    if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
        raise RuntimeError(
            "CRITICAL SECURITY ERROR: liboqs not found in PRODUCTION_MODE. "
            "Cannot use insecure fallback for encapsulation."
        )

    logger.warning("SECURITY: Using SHAKE-256 Kyber encapsulation simulation (NOT secure)")

    # Generate a random seed and embed it in the ciphertext so the decapsulator
    # can derive the same shared secret. This is NOT how real Kyber works but
    # ensures encaps/decaps are consistent in the fallback path.
    randomness = secrets.token_bytes(32)

    # Shared secret = SHAKE-256(public_key || randomness || "kyber_ss_v2")
    ss_shake = hashlib.shake_256()
    ss_shake.update(public_key)
    ss_shake.update(randomness)
    ss_shake.update(b"kyber_ss_v2")
    shared_secret = ss_shake.digest(sizes["shared_secret"])

    # Ciphertext = randomness || SHAKE-256 padding to reach expected ct size
    # The randomness is needed by decapsulate to re-derive the shared secret.
    ct_shake = hashlib.shake_256()
    ct_shake.update(randomness)
    ct_shake.update(b"kyber_ct_pad_v2")
    ct_padding = ct_shake.digest(sizes["ciphertext"] - 32)
    ciphertext = randomness + ct_padding

    return ciphertext, shared_secret


def decapsulate_kyber(
    ciphertext: bytes,
    secret_key: bytes,
    level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
) -> bytes:
    """
    Decapsulate to derive shared secret.

    Uses liboqs when available; falls back to SHAKE-256 simulation.

    Args:
        ciphertext: Ciphertext from encapsulation
        secret_key: Recipient's secret key
        level: Security level

    Returns:
        Shared secret (32 bytes)
    """
    sizes = KYBER_SIZES[level]

    if len(ciphertext) != sizes["ciphertext"]:
        raise ValueError(f"Invalid ciphertext size for Kyber{level.value}")
    if len(secret_key) != sizes["secret_key"]:
        raise ValueError(f"Invalid secret key size for Kyber{level.value}")

    if _LIBOQS_AVAILABLE:
        kem = oqs.KeyEncapsulation(_KYBER_OQS_NAMES[level], secret_key)
        shared_secret = kem.decap_secret(ciphertext)
        return bytes(shared_secret)

    # Fallback: extract randomness from ciphertext and re-derive shared secret (PY-02 fix)
    if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
        raise RuntimeError(
            "CRITICAL SECURITY ERROR: liboqs not found in PRODUCTION_MODE. "
            "Cannot use insecure fallback for decapsulation."
        )

    logger.warning("SECURITY: Using SHAKE-256 Kyber decapsulation simulation (NOT secure)")

    # Extract the 32-byte randomness seed from the front of the ciphertext
    # (embedded by encapsulate_kyber fallback).
    randomness = ciphertext[:32]

    # The public key is stored in the last pk_size bytes of the secret key
    # (embedded during keygen, similar to how real Kyber sk includes pk).
    pk_size = sizes["public_key"]
    public_key = secret_key[-pk_size:]

    # Re-derive shared secret using the same formula as encapsulate
    ss_shake = hashlib.shake_256()
    ss_shake.update(public_key)
    ss_shake.update(randomness)
    ss_shake.update(b"kyber_ss_v2")
    return ss_shake.digest(sizes["shared_secret"])


# ============ Utility Functions ============


def kyber_key_sizes(level: KyberSecurityLevel) -> dict[str, int]:
    """
    Get key, ciphertext, and shared secret sizes for a security level.

    Args:
        level: Security level

    Returns:
        Dictionary with sizes
    """
    return KYBER_SIZES[level].copy()


def is_valid_kyber_public_key(
    public_key: bytes,
    level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
) -> bool:
    """Check if bytes are a valid Kyber public key."""
    return len(public_key) == KYBER_SIZES[level]["public_key"]


def is_valid_kyber_ciphertext(
    ciphertext: bytes,
    level: KyberSecurityLevel = KyberSecurityLevel.LEVEL3,
) -> bool:
    """Check if bytes are a valid Kyber ciphertext."""
    return len(ciphertext) == KYBER_SIZES[level]["ciphertext"]


# ============ Hybrid Key Exchange ============


def hybrid_key_exchange(
    classical_shared_secret: bytes,
    pqc_shared_secret: bytes,
) -> bytes:
    """
    Combine classical and post-quantum shared secrets.

    For defense in depth, combine ECDH shared secret with Kyber
    shared secret using a KDF.

    Args:
        classical_shared_secret: ECDH shared secret
        pqc_shared_secret: Kyber shared secret

    Returns:
        Combined 32-byte shared secret
    """
    shake = hashlib.shake_256()
    shake.update(b"aethelred_hybrid_kex")
    shake.update(len(classical_shared_secret).to_bytes(4, "big"))
    shake.update(classical_shared_secret)
    shake.update(len(pqc_shared_secret).to_bytes(4, "big"))
    shake.update(pqc_shared_secret)

    return shake.digest(32)
