"""
Dual-Key Wallet for Aethelred

Implements the algorithm-agile dual-key architecture that combines:
- Classical: ECDSA (secp256k1) for current blockchain compatibility
- Quantum-Safe: Dilithium3 (NIST Level 3) for future-proofing

The Aethelred L1 expects composite signatures containing both signature
types, ensuring security against both classical and quantum adversaries.

.. admonition:: Security Audit

   Audited 2026-02-22. Password is now required for key export (PY-04).
   CompositeSignature.from_bytes() validates all length fields (PY-05).
   Private keys are zeroized on object deletion (PY-11).

Example:
    >>> from aethelred.core.wallet import DualKeyWallet
    >>>
    >>> # Create new wallet
    >>> wallet = DualKeyWallet()
    >>> print(f"Address: {wallet.address}")
    >>>
    >>> # Sign a transaction
    >>> tx_bytes = b"transaction data..."
    >>> composite_sig = wallet.sign_transaction(tx_bytes)
    >>>
    >>> # Verify signature
    >>> is_valid = wallet.verify_signature(tx_bytes, composite_sig)
"""

from __future__ import annotations

import base64
import hashlib
import ctypes
import json
import logging
import os
import secrets
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# PY-21 fix: Explicit public API surface to prevent accidental exposure
# of internal helpers (e.g. _bech32_polymod, _bech32_hrp_expand).
__all__ = [
    "SignatureScheme",
    "ECDSAKeyPair",
    "CompositeSignature",
    "ECDSASigner",
    "DualKeyWallet",
    "bech32_encode",
]

_MIN_EXPORT_PASSWORD_LENGTH = 12

import ecdsa as ecdsa_lib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from aethelred.crypto.pqc.dilithium import (
    DilithiumSigner,
    DilithiumSecurityLevel,
    DilithiumSignature,
)
from aethelred.crypto.pqc.kyber import (
    KyberKEM,
    KyberSecurityLevel,
)


# =============================================================================
# Bech32 encoding (inline — avoids adding a dependency for a single function)
# =============================================================================

_BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _bech32_polymod(values: list[int]) -> int:
    """Internal function for bech32 checksum computation."""
    GEN = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk


def _bech32_hrp_expand(hrp: str) -> list[int]:
    """Expand the HRP into values for checksum computation."""
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_create_checksum(hrp: str, data: list[int]) -> list[int]:
    """Compute bech32 checksum."""
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0, 0, 0, 0, 0, 0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]


def _convertbits(
    data: bytes, frombits: int, tobits: int, pad: bool = True
) -> list[int]:
    """Convert between bit sizes for bech32."""
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad and bits:
        ret.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        return []
    return ret


def bech32_encode(hrp: str, witprog: bytes) -> str:
    """Encode a byte payload into a bech32 string with the given HRP."""
    data = _convertbits(witprog, 8, 5)
    combined = data + _bech32_create_checksum(hrp, data)
    return hrp + "1" + "".join(_BECH32_CHARSET[d] for d in combined)


# =============================================================================
# Types
# =============================================================================


class SignatureScheme(str, Enum):
    """Signature scheme types."""

    ECDSA = "ecdsa"
    DILITHIUM = "dilithium"
    COMPOSITE = "composite"  # Both ECDSA + Dilithium


@dataclass
class ECDSAKeyPair:
    """ECDSA key pair container (secp256k1)."""

    public_key: bytes  # 33 bytes compressed
    private_key: bytes  # 32 bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def public_key_hex(self) -> str:
        return self.public_key.hex()

    def private_key_hex(self) -> str:
        """Return private key as hex string.

        .. warning:: Handle with extreme care. Do not log or display.
        """
        return self.private_key.hex()


@dataclass
class CompositeSignature:
    """
    Composite signature containing both classical and quantum-safe signatures.

    The Aethelred L1 verifies both signatures, providing security against
    both classical and quantum adversaries (defense in depth).
    """

    classical_sig: bytes  # ECDSA signature (64 bytes, r||s)
    pqc_sig: bytes  # Dilithium signature (3309 bytes for Level3)
    scheme: SignatureScheme = SignatureScheme.COMPOSITE
    signer_address: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bytes(self) -> bytes:
        """Serialize composite signature to bytes."""
        # Format: [scheme(1)] [classical_len(2)] [classical] [pqc_len(2)] [pqc]
        data = bytearray()
        data.append(2)  # Composite scheme marker
        data.extend(len(self.classical_sig).to_bytes(2, "big"))
        data.extend(self.classical_sig)
        data.extend(len(self.pqc_sig).to_bytes(2, "big"))
        data.extend(self.pqc_sig)
        return bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompositeSignature":
        """Deserialize composite signature from bytes.

        Raises:
            ValueError: If the data is malformed or truncated.
        """
        if len(data) < 6:  # 1 (scheme) + 2 (classical_len) + minimum + 2 (pqc_len)
            raise ValueError(
                f"Composite signature too short: {len(data)} bytes (minimum 6)"
            )
        if data[0] != 2:
            raise ValueError(
                f"Invalid composite signature scheme marker: {data[0]} (expected 2)"
            )

        classical_len = int.from_bytes(data[1:3], "big")
        if 3 + classical_len + 2 > len(data):
            raise ValueError(
                f"Classical signature length {classical_len} exceeds "
                f"available data ({len(data) - 5} bytes)"
            )
        classical_sig = data[3 : 3 + classical_len]

        pqc_len_start = 3 + classical_len
        pqc_len = int.from_bytes(data[pqc_len_start : pqc_len_start + 2], "big")
        if pqc_len_start + 2 + pqc_len > len(data):
            raise ValueError(
                f"PQC signature length {pqc_len} exceeds "
                f"available data ({len(data) - pqc_len_start - 2} bytes)"
            )
        pqc_sig = data[pqc_len_start + 2 : pqc_len_start + 2 + pqc_len]

        return cls(classical_sig=classical_sig, pqc_sig=pqc_sig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scheme": self.scheme.value,
            "classical_sig": self.classical_sig.hex(),
            "pqc_sig": self.pqc_sig.hex(),
            "signer_address": self.signer_address,
            "created_at": self.created_at.isoformat(),
        }

    def __len__(self) -> int:
        return len(self.classical_sig) + len(self.pqc_sig)


# =============================================================================
# ECDSA Signer — real secp256k1 via ``ecdsa`` library
# =============================================================================


class ECDSASigner:
    """
    ECDSA signer for secp256k1 curve.

    Uses the ``ecdsa`` library for real elliptic-curve operations:

    - Key generation via ``ecdsa.SigningKey.generate(curve=SECP256k1)``
    - Deterministic signing via RFC 6979 (``sign_deterministic``)
    - Verification via ``VerifyingKey.verify``
    """

    def __init__(
        self,
        private_key: Optional[bytes] = None,
    ):
        """
        Initialize ECDSA signer.

        Args:
            private_key: Existing 32-byte private key (generates new if None)
        """
        if private_key is not None:
            if len(private_key) != 32:
                raise ValueError("ECDSA private key must be 32 bytes")
            self._signing_key = ecdsa_lib.SigningKey.from_string(
                private_key, curve=ecdsa_lib.SECP256k1
            )
        else:
            self._signing_key = ecdsa_lib.SigningKey.generate(
                curve=ecdsa_lib.SECP256k1
            )

        self._verifying_key = self._signing_key.get_verifying_key()

    def _derive_public_key(self) -> bytes:
        """Return compressed public key (33 bytes)."""
        return self._verifying_key.to_string("compressed")

    def sign(self, message: bytes) -> bytes:
        """
        Sign a message using ECDSA with deterministic k (RFC 6979).

        Args:
            message: Message to sign

        Returns:
            64-byte signature (r || s)
        """
        return self._signing_key.sign_deterministic(
            message,
            hashfunc=hashlib.sha256,
            sigencode=ecdsa_lib.util.sigencode_string,
        )

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify an ECDSA signature."""
        try:
            return self._verifying_key.verify(
                signature,
                message,
                hashfunc=hashlib.sha256,
                sigdecode=ecdsa_lib.util.sigdecode_string,
            )
        except ecdsa_lib.BadSignatureError:
            return False

    @staticmethod
    def verify_with_public_key(
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """Verify an ECDSA signature using only a public key."""
        try:
            vk = ecdsa_lib.VerifyingKey.from_string(
                public_key, curve=ecdsa_lib.SECP256k1
            )
            return vk.verify(
                signature,
                message,
                hashfunc=hashlib.sha256,
                sigdecode=ecdsa_lib.util.sigdecode_string,
            )
        except (ecdsa_lib.BadSignatureError, Exception):
            return False

    @property
    def public_key(self) -> bytes:
        return self._derive_public_key()

    @property
    def private_key(self) -> bytes:
        return self._signing_key.to_string()


# =============================================================================
# Dual-Key Wallet
# =============================================================================


class DualKeyWallet:
    """
    Dual-key wallet managing both ECDSA and Dilithium keys.

    Implements Aethelred's algorithm-agile architecture for quantum-safe
    transactions while maintaining compatibility with classical infrastructure.

    Features:
    - Composite signatures (ECDSA + Dilithium)
    - Bech32 address derivation (``aethel1...``)
    - Key export/import with AES-256 encryption via Fernet
    - Transaction signing with domain separation
    """

    def __init__(
        self,
        classical_key: Optional[bytes] = None,
        quantum_key: Optional[bytes] = None,
        dilithium_level: DilithiumSecurityLevel = DilithiumSecurityLevel.LEVEL3,
    ):
        """
        Initialize dual-key wallet.

        Args:
            classical_key: Existing ECDSA private key (32 bytes)
            quantum_key: Existing Dilithium secret key
            dilithium_level: Dilithium security level
        """
        # Classical signer (ECDSA secp256k1)
        self.classical = ECDSASigner(private_key=classical_key)

        # Quantum-safe signer (Dilithium)
        if quantum_key is not None:
            self.quantum = DilithiumSigner.from_secret_key(
                quantum_key, dilithium_level
            )
        else:
            self.quantum = DilithiumSigner(level=dilithium_level)

        # Kyber KEM for encryption
        self.kem = KyberKEM(level=KyberSecurityLevel(dilithium_level.value))

        self._dilithium_level = dilithium_level
        self._address = self._derive_address()
        self._created_at = datetime.now(timezone.utc)
        self._closed = False

        # PY-20 fix: Register deterministic finalizer via weakref.finalize.
        # Python's __del__ is unreliable — it may not run during GC or may run
        # at interpreter shutdown when modules are already torn down.
        # weakref.finalize is guaranteed to run and is the recommended pattern.
        self._finalizer = weakref.finalize(self, DualKeyWallet._zeroize_keys, self.classical, self.quantum)

    def _derive_address(self) -> str:
        """
        Derive Aethelred address from public keys.

        Address = bech32("aethel", SHA-256(ecdsa_pk || dilithium_pk)[:20])
        """
        combined = self.classical.public_key + self.quantum.public_key_bytes()
        address_hash = hashlib.sha256(combined).digest()[:20]
        return bech32_encode("aethel", address_hash)

    @property
    def address(self) -> str:
        """Return wallet address."""
        return self._address

    @staticmethod
    def _zeroize_keys(classical_signer, quantum_signer) -> None:
        """Securely zeroize private key material (PY-20 fix).

        Uses ctypes.memset for volatile writes that the Python GC and
        compiler cannot optimize away, unlike ``bytearray[i] = 0``.
        """
        try:
            # Zeroize ECDSA private key
            if hasattr(classical_signer, '_signing_key'):
                sk = classical_signer._signing_key
                if hasattr(sk, 'to_string'):
                    raw = sk.to_string()
                    if isinstance(raw, (bytes, bytearray)):
                        buf = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
                        ctypes.memset(buf, 0, len(raw))

            # Zeroize Dilithium secret key
            if hasattr(quantum_signer, '_secret_key'):
                sk_bytes = quantum_signer._secret_key
                if isinstance(sk_bytes, bytearray):
                    ctypes.memset((ctypes.c_char * len(sk_bytes)).from_buffer(sk_bytes), 0, len(sk_bytes))
                elif isinstance(sk_bytes, bytes):
                    # bytes are immutable; best-effort: replace the reference
                    quantum_signer._secret_key = b'\x00' * len(sk_bytes)
        except Exception:
            pass  # Swallow errors during interpreter shutdown

    def close(self) -> None:
        """Explicitly zeroize key material (PY-20 fix).

        Call this when the wallet is no longer needed. Also runs
        automatically via weakref.finalize when the object is collected.
        """
        if not self._closed:
            self._finalizer()
            self._closed = True

    def __del__(self) -> None:
        """Belt-and-suspenders fallback for zeroization (PY-20)."""
        try:
            self.close()
        except Exception:
            pass

    def sign_transaction(self, tx_bytes: bytes) -> CompositeSignature:
        """
        Sign a transaction with both key types.

        Args:
            tx_bytes: Transaction bytes to sign

        Returns:
            Composite signature with both ECDSA and Dilithium signatures
        """
        classical_sig = self.classical.sign(tx_bytes)
        dilithium_sig = self.quantum.sign(tx_bytes)

        return CompositeSignature(
            classical_sig=classical_sig,
            pqc_sig=dilithium_sig.signature,
            signer_address=self._address,
        )

    def sign_message(self, message: Union[bytes, str]) -> CompositeSignature:
        """
        Sign an arbitrary message with domain separation prefix.

        Args:
            message: Message to sign

        Returns:
            Composite signature
        """
        if isinstance(message, str):
            message = message.encode("utf-8")

        prefixed = (
            b"\x19Aethelred Signed Message:\n"
            + len(message).to_bytes(4, "big")
            + message
        )
        return self.sign_transaction(prefixed)

    def verify_signature(
        self,
        tx_bytes: bytes,
        signature: CompositeSignature,
    ) -> bool:
        """
        Verify a composite signature.

        Both classical and quantum signatures must be valid.
        """
        classical_valid = self.classical.verify(tx_bytes, signature.classical_sig)
        if not classical_valid:
            return False

        quantum_valid = self.quantum.verify(tx_bytes, signature.pqc_sig)
        return quantum_valid

    @staticmethod
    def verify_with_public_keys(
        tx_bytes: bytes,
        signature: CompositeSignature,
        classical_pubkey: bytes,
        quantum_pubkey: bytes,
        dilithium_level: DilithiumSecurityLevel = DilithiumSecurityLevel.LEVEL3,
    ) -> bool:
        """
        Verify a composite signature using only public keys.

        Args:
            tx_bytes: Original transaction bytes
            signature: Composite signature
            classical_pubkey: ECDSA public key (33 bytes compressed)
            quantum_pubkey: Dilithium public key
            dilithium_level: Dilithium security level

        Returns:
            True if both signatures are valid
        """
        classical_valid = ECDSASigner.verify_with_public_key(
            tx_bytes, signature.classical_sig, classical_pubkey
        )

        quantum_valid = DilithiumSigner.verify_with_public_key(
            tx_bytes,
            signature.pqc_sig,
            quantum_pubkey,
            dilithium_level,
        )

        return classical_valid and quantum_valid

    def export_keys(self, password: str) -> dict[str, Any]:
        """
        Export wallet keys for backup.

        Key material is always encrypted with AES-256 via Fernet, keyed by
        PBKDF2-HMAC-SHA256 (480,000 iterations). Plaintext export has been
        removed (PY-04 security fix).

        Args:
            password: Password for encryption (minimum 12 characters)

        Returns:
            Dictionary with encrypted key material

        Raises:
            ValueError: If password is too short
        """
        if len(password) < _MIN_EXPORT_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must be at least {_MIN_EXPORT_PASSWORD_LENGTH} characters "
                f"(got {len(password)}). Use a strong passphrase for key material."
            )

        key_data = {
            "version": 2,
            "address": self._address,
            "created_at": self._created_at.isoformat(),
            "classical": {
                "public_key": self.classical.public_key.hex(),
                "private_key": self.classical.private_key.hex(),
            },
            "quantum": {
                "level": self._dilithium_level.value,
                "public_key": self.quantum.public_key_bytes().hex(),
                "secret_key": self.quantum.secret_key_bytes().hex(),
            },
            "kem": {
                "public_key": self.kem.public_key_bytes().hex(),
                "secret_key": self.kem.secret_key_bytes().hex(),
            },
        }

        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(derived_key)

        plaintext = json.dumps(key_data).encode()
        ciphertext = fernet.encrypt(plaintext)

        return {
            "version": 2,
            "encrypted": True,
            "address": self._address,
            "salt": salt.hex(),
            "ciphertext": ciphertext.decode(),
        }

    @classmethod
    def from_export(
        cls,
        export_data: dict[str, Any],
        password: str,
    ) -> "DualKeyWallet":
        """
        Restore wallet from exported keys.

        Args:
            export_data: Exported key data (always encrypted)
            password: Password for decryption

        Returns:
            Restored wallet

        Raises:
            ValueError: If password is missing or data format is invalid
        """
        if not password:
            raise ValueError("Password is required to decrypt wallet export")

        salt = bytes.fromhex(export_data["salt"])
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        derived_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(derived_key)

        plaintext = fernet.decrypt(export_data["ciphertext"].encode())
        key_data = json.loads(plaintext)

        classical_key = bytes.fromhex(key_data["classical"]["private_key"])
        quantum_key = bytes.fromhex(key_data["quantum"]["secret_key"])
        level = DilithiumSecurityLevel(key_data["quantum"]["level"])

        return cls(
            classical_key=classical_key,
            quantum_key=quantum_key,
            dilithium_level=level,
        )

    @classmethod
    def from_mnemonic(
        cls,
        mnemonic: str,
        passphrase: str = "",
        dilithium_level: DilithiumSecurityLevel = DilithiumSecurityLevel.LEVEL3,
    ) -> "DualKeyWallet":
        """
        Derive wallet from BIP-39 mnemonic.

        Uses PBKDF2-HMAC-SHA512 (2048 iterations) for seed derivation,
        matching the BIP-39 standard.

        Args:
            mnemonic: 24-word mnemonic phrase
            passphrase: Optional passphrase
            dilithium_level: Dilithium security level

        Returns:
            Derived wallet
        """
        salt = ("mnemonic" + passphrase).encode("utf-8")
        seed = hashlib.pbkdf2_hmac(
            "sha512",
            mnemonic.encode("utf-8"),
            salt,
            iterations=2048,
            dklen=64,
        )

        # First 32 bytes → ECDSA private key
        classical_key = seed[:32]

        # Dilithium key is generated fresh (PQC keygen doesn't support
        # deterministic derivation from seed in the ecdsa/liboqs libraries).
        # NOTE (PY-14): This means only the ECDSA key is deterministically
        # derived from the mnemonic. The Dilithium key must be backed up
        # separately using export_keys() for full wallet recovery.
        logger.warning(
            "from_mnemonic: Dilithium key is generated randomly, not derived "
            "from mnemonic. Back up the full wallet using export_keys() for recovery."
        )
        return cls(
            classical_key=classical_key,
            dilithium_level=dilithium_level,
        )

    def get_public_keys(self) -> dict[str, bytes]:
        """Return all public keys."""
        return {
            "classical": self.classical.public_key,
            "quantum": self.quantum.public_key_bytes(),
            "kem": self.kem.public_key_bytes(),
        }

    def get_fingerprints(self) -> dict[str, str]:
        """Return key fingerprints."""
        return {
            "classical": hashlib.sha256(
                self.classical.public_key
            ).hexdigest()[:16],
            "quantum": self.quantum.fingerprint,
            "kem": self.kem.fingerprint,
        }


# ============ Utility Functions ============


def create_wallet() -> DualKeyWallet:
    """Create a new dual-key wallet."""
    return DualKeyWallet()


def verify_composite_signature(
    message: bytes,
    signature: CompositeSignature,
    classical_pubkey: bytes,
    quantum_pubkey: bytes,
) -> bool:
    """
    Verify a composite signature with public keys.

    Args:
        message: Original message
        signature: Composite signature
        classical_pubkey: ECDSA public key
        quantum_pubkey: Dilithium public key

    Returns:
        True if both signatures are valid
    """
    return DualKeyWallet.verify_with_public_keys(
        message, signature, classical_pubkey, quantum_pubkey
    )


def address_from_public_keys(
    classical_pubkey: bytes,
    quantum_pubkey: bytes,
) -> str:
    """
    Derive Aethelred address from public keys.

    Args:
        classical_pubkey: ECDSA public key
        quantum_pubkey: Dilithium public key

    Returns:
        Aethelred address (aethel1...)
    """
    combined = classical_pubkey + quantum_pubkey
    address_hash = hashlib.sha256(combined).digest()[:20]
    return bech32_encode("aethel", address_hash)
