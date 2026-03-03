"""
Aethelred Proof Generation and Verification Module

Tools for generating zkML proofs and verifying TEE attestations.
"""

from aethelred.proofs.generator import (
    ProofGenerator,
    ProofConfig,
    ProofRequest,
    ProofResult,
)
from aethelred.proofs.verifier import (
    ProofVerifier,
    VerificationResult,
    TEEVerifier,
    ZKVerifier,
)

__all__ = [
    # Generator
    "ProofGenerator",
    "ProofConfig",
    "ProofRequest",
    "ProofResult",
    # Verifier
    "ProofVerifier",
    "VerificationResult",
    "TEEVerifier",
    "ZKVerifier",
]
