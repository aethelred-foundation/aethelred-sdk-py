"""
Post-Quantum Cryptography (PQC) Module

NIST-approved post-quantum cryptographic algorithms for Aethelred's
algorithm-agile architecture. Provides quantum-resistant signatures
(Dilithium) and key encapsulation (Kyber) for future-proof security.

The Aethelred L1 uses a dual-signature scheme:
- Classical: ECDSA (secp256k1) for current compatibility
- Quantum-Safe: Dilithium3 (NIST Level 3) for future-proofing

Example:
    >>> from aethelred.crypto.pqc import DilithiumSigner, KyberKEM
    >>>
    >>> # Generate quantum-safe keypair
    >>> signer = DilithiumSigner()
    >>> signature = signer.sign(message)
    >>> assert signer.verify(message, signature)
    >>>
    >>> # Key encapsulation
    >>> kem = KyberKEM()
    >>> ciphertext, shared_secret = kem.encapsulate(recipient_public_key)
"""

from aethelred.crypto.pqc.dilithium import (
    DilithiumSigner,
    DilithiumKeyPair,
    DilithiumSignature,
    DilithiumSecurityLevel,
    generate_dilithium_keypair,
    sign_dilithium,
    verify_dilithium,
)
from aethelred.crypto.pqc.kyber import (
    KyberKEM,
    KyberKeyPair,
    KyberCiphertext,
    KyberSecurityLevel,
    generate_kyber_keypair,
    encapsulate_kyber,
    decapsulate_kyber,
)

__all__ = [
    # Dilithium (Signatures)
    "DilithiumSigner",
    "DilithiumKeyPair",
    "DilithiumSignature",
    "DilithiumSecurityLevel",
    "generate_dilithium_keypair",
    "sign_dilithium",
    "verify_dilithium",
    # Kyber (KEM)
    "KyberKEM",
    "KyberKeyPair",
    "KyberCiphertext",
    "KyberSecurityLevel",
    "generate_kyber_keypair",
    "encapsulate_kyber",
    "decapsulate_kyber",
]
