"""Cryptography module for Aethelred SDK.

Provides quantum-safe and classical cryptographic primitives.

Example::

    from aethelred.crypto import HybridSigner, DilithiumSigner, KyberKEM

    signer = HybridSigner()
    sig = signer.sign(b"data")
    assert signer.verify(b"data", sig)
"""

from aethelred.crypto.fallback import HybridSigner, HybridVerifier, get_backend
from aethelred.crypto.pqc.dilithium import (
    DilithiumSigner,
    DilithiumSecurityLevel,
    DilithiumSignature,
)
from aethelred.crypto.pqc.kyber import KyberKEM, KyberSecurityLevel

__all__ = [
    "HybridSigner",
    "HybridVerifier",
    "get_backend",
    "DilithiumSigner",
    "DilithiumSecurityLevel",
    "DilithiumSignature",
    "KyberKEM",
    "KyberSecurityLevel",
]
